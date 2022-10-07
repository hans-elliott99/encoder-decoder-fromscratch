# Text preprocessing
from pickle import NONE
import string
import random

# PyTorch
import torch


def import_data(file_path, max_english_chars, max_french_chars, n_samples=None):
    english, french = [],[]

    matches = ["(", "‽", "…", "€"]
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            line = line.rstrip()
            line = line.replace(u"\u202f", " ")
            line = line.replace(u"\u3000", " ")
            line = line.replace(u"\u2000", " ")
            line = line.replace(u"\u200b", " ")
            line = line.replace(u"\xa0", " ")
            line = line.replace(u"\xad", " ")
            line = line.replace(u"\u2009", " ")
            line = line.replace("ú", "u")
            line = line.replace("–", "-")
            line = line.replace("а", "a")
            line = line.replace("‐", "-")
            line = line.replace("₂", "2")
            line = line.replace("\'", "'")
            if any(s in line for s in matches): ##removes some edge cases
                pass
            else:        
                eng, fra = line.split('\t')
            
            if (len(fra)>max_french_chars) | (len(eng)>max_english_chars):
                pass
            else:
                english.append(eng)
                french.append(fra)
            if n_samples is not None: ##check if we should keep adding
                if len(english) == n_samples:
                    break

    return english, french


def words_to_ch_data(eng, fra, stoi, stop_char:str, max_Xs:int, max_Ys:int, device):
    """
    Converts words to character-level data.
    """
    assert len(eng)==len(fra)
    max_Xs = len(max(eng, key=len))
    max_Ys = len(max(fra, key=len))
    # Data
    ## block_size = context length: how many chars do we use to predict the next?
    X, Y = [], []
    for en, fr in zip(eng, fra):

        add_to_en = max_Xs-len(en) ##pad to even length
        add_to_fr = max_Ys-len(fr)

        english_ix, french_ix = [], []
        for ch in en:
            english_ix.append(stoi[ch])
        english_ix.append(stoi[stop_char])
        english_ix += [stoi[stop_char]] * add_to_en

        for ch in fr:
            french_ix.append(stoi[ch])
        french_ix.append(stoi[stop_char])
        french_ix += [stoi[stop_char]] * add_to_fr

        X.append(english_ix)
        Y.append(french_ix)

    return torch.tensor(X, device=device), torch.tensor(Y, device=device)


def words_to_word_data(eng, fra, stoi, stop_char:str, device, keep_punct=""):
    """
    Converts words to word-level data.
    """
    assert len(eng)==len(fra), f"len(eng) {len(eng)} != len(fra) {len(fra)}"

    #unpack stoi dict
    assert len(stoi.keys()) == 2, "the stoi_mapping dict must contain 2 dictionaries, the first for english words and the second for french."
    en_stoi = stoi[[k for k in stoi.keys()][0]]
    fr_stoi = stoi[[k for k in stoi.keys()][1]] 

    #prep
    remove_punct = ''.join([p for p in string.punctuation if p not in keep_punct]) ##remove all punct except...
    max_Xs = len(max([e.split() for e in eng], key=len)) ##longest english phrase
    max_Ys = len(max([f.split() for f in fra], key=len)) ##longest french phrase

    # Data
    X, Y = [], []
    for en, fr in zip(eng, fra):
        # remove unwanted punctuation
        en = ''.join([c for c in en if c not in remove_punct]).lower()
        fr = ''.join([c for c in fr if c not in remove_punct]).lower()

        # determine padding for equal length tensors
        add_to_en = max_Xs-len(en.split()) ##pad to even length
        add_to_fr = max_Ys-len(fr.split())

        # split phrases into words
        english_ix, french_ix = [], []
        for wd in en.split():
            english_ix.append(en_stoi[wd])
        english_ix.append(en_stoi[stop_char])
        english_ix += [en_stoi[stop_char]] * add_to_en

        for wd in fr.split():
            french_ix.append(fr_stoi[wd])
        french_ix.append(fr_stoi[stop_char])
        french_ix += [fr_stoi[stop_char]] * add_to_fr

        X.append(english_ix)
        Y.append(french_ix)

    return torch.tensor(X, device=device), torch.tensor(Y, device=device)



def split_samples(inputs, labels, frac=0.8, seed=123):
    """
    Split x and y tensors (inputs and labels) into train and test sets\n
    returns train_x, train_y, test_x, test_y
    """
    
    assert len(inputs)==len(labels), f"len(inputs) {len(inputs)} does not match len(labels) {len(labels)}"
    # generate a list of indices to exclude. Turn in into a set for O(1) lookup time
    random.seed(seed)
    indx = list(set(random.sample(list(range(len(inputs))), int(frac*len(inputs)))))

    x_mask = torch.zeros((len(inputs)), dtype=torch.bool) #False
    x_mask[indx] = True

    y_mask = torch.zeros((len(inputs)), dtype=torch.bool) #False
    y_mask[indx] = True

    train_x = inputs[x_mask]
    train_y = labels[y_mask]

    test_x = inputs[~x_mask]
    test_y = labels[~y_mask]

    return train_x, train_y, test_x, test_y