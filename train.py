import string
import time
import random
import pickle 

import torch


from enc_dec.model import TranslGRU
from enc_dec.data import import_data, words_to_word_data, words_to_ch_data


MAX_CHARS = 30
START_CHR = '>'
STOP_CHR = '<'
KEEP_PUNCT = "'-"
N_SAMPLES = 1000

SAVE_MODEL=True
SAVE_GRADS=True


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device = ", device)


def importPrepDicts(data_path, level="word"):
    """
    Import data and use it to prepare dictionaries mapping the character/word encodings\n
    path to data -> english phrases (list), french phrases (list), stoi (dict), itos (dict)

    data_path -- path to text file\n
    level -- level at which to break down data. either "word" or "character".\n
    """
    english, french = import_data(data_path,
                                max_english_chars=MAX_CHARS, max_french_chars=MAX_CHARS)
    
    if level=="word":
        remove_punct = ''.join([p for p in string.punctuation if p not in KEEP_PUNCT])
        # join all phrases with space in between
        all_words = ' '.join(english + french)
        # remove punctuation characters except for those not in remove_punct
        all_words = ''.join([c for c in all_words if c not in remove_punct])

        # create mapping from string to int
        stoi = {word:i+2 for i, word in enumerate( sorted(set(all_words.lower().split())) )}
        stoi[START_CHR] = 1
        stoi[STOP_CHR] = 0

        itos = {i:w for w, i in stoi.items()}

    if level=="character":
        all_chars = set(''.join(english + french))

        stoi = {s:i+2 for i, s in enumerate(all_chars)} ##create dictionary mapping from char to int
        stoi[START_CHR] = 1
        stoi[STOP_CHR] = 0
        itos = {i:s for s, i in stoi.items()}

    return english, french, stoi, itos


def train(hidden_size, enc_embedding_dim, dec_embedding_dim, max_length=100, teacher_forcing=0.0, lr=3e-4, epochs=10, print_every=1):
    """Main: Pull in data and train model for desired epochs. Saves epoch-loss to log file and pickles final weight tensors."""
    # IMPORT AND PREP DATA
    english, french, stoi, itos = importPrepDicts("./data/eng-fra.txt", level="word")
    
    Xs, Ys = words_to_word_data(english[:N_SAMPLES], french[:N_SAMPLES], 
                                stoi, stop_char=STOP_CHR, 
                                device=device,
                                keep_punct=KEEP_PUNCT)

    n_words = len(stoi)

    # INITIALIZE MODEL
    model = TranslGRU(vocab_size=n_words,
                    enc_embed_dim=enc_embedding_dim,
                    dec_embed_dim=dec_embedding_dim,
                    dec_type="attention",
                    hidden_size=hidden_size,
                    max_length=max_length,
                    stoi_mapping=stoi, 
                    start_chr=START_CHR, stop_chr=STOP_CHR
                    )
    model.init_weights(device=device)

    # PRINT MODEL 
    for n, p in zip(model.param_names, model.params):
        print(f"| {n}  | shape = {p.shape[0], p.shape[1]} | params =  {p.nelement():,} | {p.device}")

    print(f"\ntotal params: {model.n_parameters:,}")

    
    # TRAINING LOOP
    optimizer = torch.optim.Adam(model.params, lr=lr)

    plot_every = 1
    log_file = './data/train-log.txt'
    loss_list = []

    open(log_file, 'w').close() #empty the log file if it exists


    assert (Xs[0].device == device) & (Ys[0].device == device), f"put data on {device}"
    print("beginning training... \n")

    strt = time.time()
    for epoch in range(1, epochs+1):
        ep_loss = 0
        sample_ix = [i for i in range(0, N_SAMPLES)]
        random.shuffle(sample_ix)

        # iterate through every sample in the training data
        for sample in sample_ix:
            x, y = Xs[sample], Ys[sample] ##already on device
            output, loss = model.forward(x, y, teacher_forcing_ratio=teacher_forcing)
            ep_loss += loss.item()

            model.backprop_update(optimizer)
        et = time.time() ##elapsed time

        # write epoch loss to log
        with open(log_file, 'a') as f:
            f.write(f'epoch {epoch} (et {round(et-strt, 3)}s); loss {ep_loss / N_SAMPLES} \n')

        # print
        if epoch % print_every == 0:
            guess, true = model.decode_output(output, label=y, spaces=True) ##use the last training sample as an example

            print(f"({epoch}/{epochs}) loss = {round(ep_loss/N_SAMPLES, 6)}", end=" ")
            print(f"(et: {round(et-strt, 3)}s)")
            print("GUESS:", guess, end="   ")
            print("TRUTH:", true, end="   ")
            print("ENG:", ' '.join([itos[i.item()] for i in x]))

        if epoch % plot_every == 0:
            loss_list.append(ep_loss/N_SAMPLES)
    
    return model


if __name__=='__main__':
    # argparse...
    model = train(hidden_size=64, 
                  enc_embedding_dim=64, dec_embedding_dim=64, 
                  max_length=100,
                  teacher_forcing=0.25, lr=3e-4,
                  epochs=10, 
                  print_every=1)


    if SAVE_MODEL:
        final_model = dict(  names = model.param_names, 
                             weights = [p.cpu() for p in model.params],
                             config = {'vocab_size': model.input_size, 
                                       'enc_embed_dim': model.enc_embed_dim, 
                                       'dec_embed_dim': model.dec_embed_dim,
                                       'dec_type' : model.dec_type,
                                       'hidden_size' : model.enc_hidden_size, 
                                       'max_length' : model.max_length,
                                       'stoi' : model.stoi, 'start_chr' : START_CHR, 'stop_chr' : STOP_CHR
                                       }
                             )
        if SAVE_GRADS:
            final_model.update(grads = [p.grad.cpu() for p in model.params])

        with open('./data/saved_model.pkl', 'wb') as f:
            pickle.dump(final_model, f)
