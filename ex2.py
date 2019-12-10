import collections
from funtiontable import FunctionTable
import functions as f
import pandas as pd
import numpy as np

import re


get_frequency_norms =True
get_pos_ratios=True
get_density=True

feature_dict = collections.defaultdict(int)
pos_features = collections.defaultdict(int)

#list_file = open('/Users/rjsgm/PycharmProjects/Covefefe_kor/output_folder/pos.txt', 'r',encoding='UTF-8').read().split('\n')


list_file=pd.read_csv('/Users/rjsgm/PycharmProjects/Covefefe_kor/output_folder/pos.txt',names=['pos','token'])



a=len(list_file.index)


pos_features['word_length'] =a


list_string_file=' '.join(list_file['pos'])
total_words=len(list_file['pos'])

a=list_file.to_dict('dict')

def get_pos_features(data):
    #기본 피처 추가
    pos_keys = ['word_length']
    pos_keys += ['nouns', 'verbs', 'inflected_verbs', 'light', 'function', 'pronouns', 'determiners', 'adverbs',
                 'adjectives', 'prepositions',
                 'coordinate', 'subordinate', 'demonstratives']
    if get_frequency_norms:
        pos_keys += ['frequency', 'noun_frequency', 'verb_frequency']
    if get_pos_ratios:
        pos_keys += ['nvratio', 'prp_ratio', 'noun_ratio', 'sub_coord_ratio']
    if get_density:
        pos_keys += ['prop_density', 'content_density']

    if get_density:
        pos_features['prop_density'] = 0
        pos_features['content_density'] =0

    ##########################################################




    for s in data.values():
        if get_density:
            if ("NNG" or "VV" or "VA" or "MAG" or "SF" or "SE" or "SSO" or "SSC" or "SC" or "SY") in a['pos'].s:
                pos_features['content_density'] += 1
            if ("VV" or "VA" or "MAG" or "MAJ") in a['pos'].s:
                pos_features['prop_density'] += 1

        if "NN" in a['pos'].s:
            pos_features['nouns'] = pos_features['nouns'] + 1

#print(list_file[['pos','token']])
#get_pos_features(list_file)
get_pos_features(a)


print(pos_features.keys())
print(collections.Counter(pos_features))


