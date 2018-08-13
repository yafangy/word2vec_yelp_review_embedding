#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 11:41:14 2018

@author: Yafang Yang
"""
import gensim
from gensim.models import word2vec
import logging

import os
import zipfile

import nltk
from nltk import word_tokenize


vector_dim = 100

# convert the input data into a list of integer indexes aligning with the wv indexes
# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = f.read(f.namelist()[0]).split()
    return data

def convert_data_to_index(string_data, wv):
    index_data = []
    for word in word_tokenize(string_data):
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data

class MySentences(object):
    def __init__(self, dirname, fname = None):
        self.dirname = dirname
        self.fname = fname
 
    def __iter__(self):
        if self.fname == None:
            for fname in os.listdir(self.dirname):
                for line in open(os.path.join(self.dirname, fname)):
                    yield word_tokenize(line)
        else:
            for line in open(os.path.join(self.dirname, self.fname)):
                yield word_tokenize(line)
                

filename = 'Review_text.txt.zip'
if not os.path.exists((root_path + filename).strip('.zip')):
    zipfile.ZipFile(root_path+filename).extractall()
sentences = MySentences(root_path, filename.strip('.zip'))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = word2vec.Word2Vec(sentences, iter=10, min_count=5, size=vector_dim, workers=4)

# save and reload the model
model.save(root_path + filename.strip('.zip').strip('.txt') + '_gensim_model')

# get the most common words
print([model.wv.index2word[i] for i in range(50)])
# some similarity fun
print(model.wv.similarity('woman', 'man'), model.wv.similarity('man', 'eggplant'))

# what doesn't fit?
print(model.wv.doesnt_match("green blue red zebra".split()))

vocab_size = len(model.wv.vocab)
print(model.wv.index2word[vocab_size - 1], model.wv.index2word[vocab_size - 2], model.wv.index2word[vocab_size - 3])
print(model ['man'])
print(model.predict_output_word(['chicken']))
str_data = read_data((root_path + filename).strip('.zip'))
index_data = convert_data_to_index(str_data, model.wv)
print(str_data[:50], index_data[:50])

# save word embedding to a text file
with open('yelp_emb.txt','w') as F:
    vocab=sorted(model.wv.vocab)
    F.write(str(len(vocab))+' '+str(vector_dim)+'\n')
    for word in vocab:
        F.write(word+' ')
        for elemt in model.wv.word_vec(word):
            F.write("%s " % elemt)
        F.write('\n')
    