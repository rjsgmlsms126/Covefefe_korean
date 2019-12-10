import collections
from funtiontable import FunctionTable
import functions as f
import pandas as pd
import numpy as np

import re


get_frequency_norms =True
get_pos_ratios=True
get_density=True
norms_freq =f.get_frequency_norms()
norms_word=list(norms_freq.keys())

feature_dict = collections.defaultdict(int)
pos_features = collections.defaultdict(int)

#list_file = open('/Users/rjsgm/PycharmProjects/Covefefe_kor/output_folder/pos.txt', 'r',encoding='UTF-8').read().split('\n')


list_file=pd.read_csv('/Users/rjsgm/PycharmProjects/Covefefe_kor/output_folder/pos.txt',names=['pos','token'])


#df 나눠주기 pos , token 으로
#print(list_file)
pos=list_file.loc[:,['pos']]
token=list_file.loc[:,['token']]


a=len(pos.index)
b=pos.to_dict('list')
c=token.to_dict('list')


for p_value in b.values():
   for i in range(len(p_value)):
        print(p_value[i])


for t_value in c.values():
   for i in range(len(t_value)):
        print(t_value[i])

import nltk
import os
import re
import functions as f

def get_frequency_norms():
    """Parameters:
    path_to_norms : optional, string. Full path, including filename, of the frequency norms.

    Return dictionary of SUBTL frequencies, keys = words, values = list of norms."""



    with open('C:/Users/rjsgm/PycharmProjects/Covefefe_kor/dic/freq_kor.txt', "r") as fin:
        f = fin.readlines()
        f = f[1:]  # skip header

        freq = {}
        for line in f:
            l = line.strip().split()
            if len(l) == 0:
                continue
            freq[l[0].lower()] = l[1:]  # returns whole line -- usually just use Log10WF

        return freq
    return None


norms_freq =f.get_frequency_norms()
print(norms_freq.keys())
norms_word=list(norms_freq.keys())
print(norms_freq.values())


norms_log10wf=list(norms_freq.values())

print(norms_word[0])
print(norms_log10wf[0][1])
print(norms_word.index('또'))
print(t_value[1])




#print(norms_log10wf)
#print(norms_word)
#print(t_value[1])
print(norms_word)

for i in norms_word:
    print(i)
fdist_vocab = nltk.probability.FreqDist([word for word in norms_word])
vocab_words = list(fdist_vocab.keys())

print(vocab_words)
print(fdist_vocab.values())
print(len(norms_word))