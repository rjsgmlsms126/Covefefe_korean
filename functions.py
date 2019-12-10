import nltk
import os
import re
import functions as f
import csv

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


def get_mpqa_lexicon():

    with open('C:/Users/rjsgm/PycharmProjects/Covefefe_kor/dic/mpqa_kor_final.csv', "r",encoding='utf-8') as f:
        reader = csv.reader(f)
        mpqa_list = list(reader)
        words =[word[1] for word in mpqa_list]
        types = [type[0] for type in mpqa_list]
        polarities = [polar[3] for polar in mpqa_list]

    return [words, types, polarities]


