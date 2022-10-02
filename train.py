import string
import time
import random
import pickle 

import torch


from model import TranslGRU
from data import import_data, words_to_word_data, words_to_ch_data


MAX_CHARS = 30
START_CHR = '>'
STOP_CHR = '<'
KEEP_PUNCT = "'-"
N_SAMPLES = 1000


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device = ", device)

def importPrepDicts(data_path, level="word"):

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
        print(f"| {n}  | {p.shape[0], p.shape[1]} | n =  {p.nelement():,} | {p.device}")

    print("\n total params:", model.n_parameters)

    
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
        
        # write epoch loss to log
        with open(log_file, 'a') as f:
            f.write(f'epoch {epoch}; loss {ep_loss / N_SAMPLES} \n')

        # print
        if epoch % print_every == 0:
            stp = time.time()
            guess, true = model.decode_output(output, label=y, spaces=True) ##use the last training sample as an example

            print(f"({epoch}/{epochs}) loss = {round(ep_loss/N_SAMPLES, 6)}", end=" ")
            print(f"(elapsed: {round(stp-strt, 3)}s)")
            print("GUESS:", guess, end="   ")
            print("TRUTH:", true, end="   ")
            print("ENG:", ' '.join([itos[i.item()] for i in x]))

        if epoch % plot_every == 0:
            loss_list.append(ep_loss/N_SAMPLES)
    
    return model


if __name__=='__main__':
    model = train(hidden_size=128, enc_embedding_dim=64, dec_embedding_dim=64, 
                    max_length=100, teacher_forcing=0.25, lr=3e-4, 
                    epochs=10, print_every=1)

    with open('./data/weights/pkl', 'wb') as f:
        pickle.dump(model.params, f)
