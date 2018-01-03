import numpy as np
import random
import string
import os
from nltk import pos_tag, word_tokenize

def init_weight(Mi,Mo):
    return np.random.random((Mi, Mo))/np.sqrt(Mi+Mo)

def all_parity_pairs(nbit):
    N = 2**nbit
    remainder = 100 - (N%100)
    Ntotal = N + remainder
    X = np.zeros((Ntotal, nbit))
    Y = np.zeros(Ntotal)
    for ii in range(Ntotal):
        i = ii % N
        for j in range(nbit):
            if(i % (2**(j+1)) != 0):
                i -= 2**j
                X[ii,j] = 1
        Y[ii] = X[ii].sum() % 2
    return X, Y

def remove_punctuation(s):
    return s.translate(string.punctuation)

def get_robert_frost():
    word2idx = {'START':0, 'END':1}
    current_idx = 2
    sentences = []
    for line in open('D:/Udemy/RNN_Udemy/robert_frost.txt'):
        line = line.strip()
        if line:
            tokens = remove_punctuation(line.lower()).split()
            sentence = []
            for t in tokens:
                if t not in word2idx:
                    word2idx[t] = current_idx
                    current_idx += 1
                idx = word2idx[t]
                sentence.append(idx)
            sentences.append(sentence)
    return sentences, word2idx

def get_tags(s):
    tuples = pos_tag(word_tokenize(s))
    return [y for x, y in tuples]

def get_poetry_classifier_data(samples_per_class, load_cached=True, save_cached=True):
    datafile = 'poetry_classifier_data.npz'
    if load_cached and os.path.exists(datafile):
        npz = np.load(datafile)
        X = npz['arr_0']
        Y = npz['arr_1']
        V = int(npz['arr_2'])
        return X, Y, V

    word2idx = {}
    current_idx = 0
    X = []
    Y = []
    for fn, label in zip(('D:/Udemy/RNN_Udemy/edgar_allan_poe.txt', 'D:/Udemy/RNN_Udemy/robert_frost.txt'), (0, 1)):
        count = 0
        for line in open(fn):
            line = line.rstrip()
            if line:
                print(line)
                # tokens = remove_punctuation(line.lower()).split()
                tokens = get_tags(line)
                if len(tokens) > 1:
                    # scan doesn't work nice here, technically could fix...
                    for token in tokens:
                        if token not in word2idx:
                            word2idx[token] = current_idx
                            current_idx += 1
                    sequence = np.array([word2idx[w] for w in tokens])
                    X.append(sequence)
                    Y.append(label)
                    count += 1
                    print(count)
                    # quit early because the tokenizer is very slow
                    if count >= samples_per_class:
                        break
    if save_cached:
        np.savez(datafile, X, Y, current_idx)
    return X, Y, current_idx
