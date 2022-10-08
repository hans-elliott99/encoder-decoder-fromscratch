import string, time, pickle, random
import torch

from enc_dec.model import TranslGRU
from enc_dec.data import import_data, words_to_word_data, split_samples
from enc_dec.utils import UpdateRatioTracker


MAX_CHARS = 30
START_CHR = '>'
STOP_CHR = '<'
KEEP_PUNCT = "'-"
N_SAMPLES = 3000

SAVE_MODEL=True
SAVE_GRADS=True
SAVE_UTILS=True


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"device = ", device)


def importPrepDicts(data_path, n_samples, level="word"):
    """
    Import data and use it to prepare dictionaries mapping the character/word encodings\n
    path to data -> english phrases (list), french phrases (list), stoi (dict), itos (dict)

    data_path -- path to text file\n
    level -- level at which to break down data. either "word" or "character".\n
    """
    english, french = import_data(data_path,
                                  max_english_chars=MAX_CHARS, max_french_chars=MAX_CHARS,
                                  n_samples=n_samples)
    
    if level=="word":
        remove_punct = ''.join([p for p in string.punctuation if p not in KEEP_PUNCT])
        # # join all phrases with space in between
        # all_words = ' '.join(english + french)
        # # remove punctuation characters except for those not in remove_punct
        # all_words = ''.join([c for c in all_words if c not in remove_punct])
        all_en, all_fr = ' '.join(english), ' '.join(french)
        all_en = ''.join([c for c in all_en if c not in remove_punct])
        all_fr = ''.join([c for c in all_fr if c not in remove_punct])
        
        # create mapping from string to int
        en_stoi = {word:i+2 for i, word in enumerate( sorted(set(all_en.lower().split())) )}
        en_stoi[START_CHR] = 1
        en_stoi[STOP_CHR] = 0

        fr_stoi = {word:i+2 for i, word in enumerate( sorted(set(all_fr.lower().split())) )}
        fr_stoi[START_CHR] = 1
        fr_stoi[STOP_CHR] = 0

        stoi = {'english':en_stoi, 'french':fr_stoi}

    if level=="character":
        all_chars = set(''.join(english + french))

        stoi = {s:i+2 for i, s in enumerate(all_chars)} ##create dictionary mapping from char to int
        stoi[START_CHR] = 1
        stoi[STOP_CHR] = 0
        itos = {i:s for s, i in stoi.items()}

    return english, french, stoi


def init_model_and_data(hidden_size, enc_embedding_dim, dec_embedding_dim, max_length=100, train_valid_split=0.8):
    """Main: Pull in data and initialize model. Split data into train and valid.\nreturns model, Xtr, Ytr, Xval, Yval"""
    # IMPORT DATA, CONVERT TO TENSOR, SPLIT INTO TRAIN & VALID
    ## Import all data in advance since it's smallish and we can avoid data loading bottleneck
    english, french, stoi = importPrepDicts("./data/eng-fra.txt", n_samples=N_SAMPLES, level="word")
    
    Xs, Ys = words_to_word_data(english, french, 
                                stoi, stop_char=STOP_CHR,
                                device=device,
                                keep_punct=KEEP_PUNCT)

    Xtr, Ytr, Xval, Yval = split_samples(Xs, Ys, frac=train_valid_split)


    # INITIALIZE MODEL
    model = TranslGRU(english_vocab_size=len(stoi['english']), french_vocab_size=len(stoi['french']),
                    enc_embed_dim=enc_embedding_dim,
                    dec_embed_dim=dec_embedding_dim,
                    dec_type="attention",
                    hidden_size=hidden_size,
                    max_length=max_length,
                    stoi_mapping=stoi, 
                    start_chr=START_CHR, stop_chr=STOP_CHR,
                    data_level="word"
                    )
    model.init_weights(device=device)

    # PRINT MODEL 
    print("__Encoder__")
    for n, p in zip(model.enc_param_names, model.enc_params):
        if 'bias' not in n.split("_"):
            print(f"| {n}  | shape = {p.shape[0], p.shape[1]} | params =  {p.nelement():,} | {p.device}")

    print("__Decoder__")
    for n, p in zip(model.dec_param_names, model.dec_params):
        if 'bias' not in n.split("_"):
            print(f"| {n}  | shape = {p.shape[0], p.shape[1]} | params =  {p.nelement():,} | {p.device}")

    print(f"\ntotal params: {model.n_parameters:,}")

    return model, Xtr, Ytr, Xval, Yval



def train(model, Xs, Ys, enc_lr=3e-4, dec_lr=3e-4, 
            teacher_forcing=0.0, 
            epochs=10, 
            print_every=1,
            valid_data:tuple=None):

    utilities = dict()

    enc_optimizer = torch.optim.Adam(model.enc_params, lr=enc_lr)
    dec_optimizer = torch.optim.Adam(model.dec_params, lr=dec_lr)

    enc_udr = UpdateRatioTracker(model.enc_params, model.enc_param_names, total_iters=Xs.shape[0], device=device, metric='std')
    dec_udr = UpdateRatioTracker(model.dec_params, model.dec_param_names, total_iters=Xs.shape[0], device=device, metric='std')

    log_file = './data/train-log.txt'
    open(log_file, 'w').close() #empty the log file if it exists

    assert (Xs[0].device == device) & (Ys[0].device == device), f"put data on {device}"
    print("beginning training... \n")

    strt = time.time()
    for epoch in range(1, epochs+1):

        # SAMPLING SETUP
        ep_loss = 0
        sample_ix = [i for i in range(0, Xs.shape[0])]
        random.shuffle(sample_ix)
         
        # iterate through every sample in the training data
        for sample in sample_ix:
            x, y = Xs[sample], Ys[sample] ##already on device
            output, loss = model.forward(x, y, teacher_forcing_ratio=teacher_forcing, max_length=100)
            ep_loss += loss.item()

            model.backprop_update(enc_optimizer, dec_optimizer)
            #metric tracking
            enc_udr.step(lr=enc_lr)
            dec_udr.step(lr=dec_lr)

        # validation data
        if valid_data is not None:
            val_loss_total=0
            x_val, y_val = valid_data
            for ix in range(0, x_val.shape[0]):
                _, val_loss = model.forward(x_val[ix], y_val[ix], teacher_forcing_ratio=0.0, max_length=100)
                val_loss_total += val_loss.item()
        val_loss = (val_loss_total / x_val.shape[0]) if valid_data is not None else None

        et = time.time() ##elapsed time
        # write epoch loss to log
        with open(log_file, 'a') as f:
            f.write(f'epoch {epoch} (et {round(et-strt, 3)}s); loss {ep_loss / N_SAMPLES}; val_loss {val_loss} \n')

        # print
        if epoch % print_every == 0:
            guess, true = model.decode_output(output, label=y, spaces=True) ##use the last training sample as an example

            print(f"({epoch}/{epochs} et: {round(et-strt, 3)}s)", end=" ")
            print(f"loss = {ep_loss/N_SAMPLES :.6f} | val_loss = {val_loss :.6f}")
            print("GUESS:", guess, end="   ")
            print("TRUTH:", true, end="   ")
            print("ENG:", ' '.join([model.en_itos[i.item()] for i in x]),end="\n\n")
    
    utilities.update(UpdateRatioTracker = {'encoder':enc_udr.output, 'decoder':dec_udr.output})
    return model, utilities


if __name__=='__main__':
    # argparse...

    model, Xtr, Ytr, Xval, Yval = init_model_and_data(hidden_size=64, enc_embedding_dim=64, dec_embedding_dim=64, 
                                                      max_length=100, train_valid_split=0.8)

    model, utilities = train(model, Xtr, Ytr,
                            enc_lr=3e-2,
                            dec_lr=3e-3,
                            teacher_forcing=0.5,
                            epochs=15, 
                            print_every=1,
                            valid_data=(Xval,Yval))


    if SAVE_MODEL:
        model_dict = dict(  names = {'encoder': model.enc_param_names, 'decoder': model.dec_param_names}, 
                             weights = {'encoder': [p.cpu() for p in model.enc_params],
                                        'decoder': [p.cpu() for p in model.dec_params]},
                             config = {'english_vocab_size': model.input_size, 
                                       'french_vocab_size': model.output_size,
                                       'enc_embed_dim': model.enc_embed_dim, 
                                       'dec_embed_dim': model.dec_embed_dim,
                                       'dec_type' : model.dec_type,
                                       'hidden_size' : model.enc_hidden_size, 
                                       'max_length' : model.max_length,
                                       'stoi' : model.stoi, 'start_chr' : START_CHR, 'stop_chr' : STOP_CHR
                                       }
                             )
        if SAVE_GRADS:
            model_dict.update(
                grads = {'encoder':[p.grad.cpu() for p in model.enc_params],
                         'decoder':[p.grad.cpu() for p in model.dec_params]}
                )
        if SAVE_UTILS:
            model_dict.update(
                utils = utilities
                    )

        with open('./data/saved_model.pkl', 'wb') as f:
            pickle.dump(model_dict, f)