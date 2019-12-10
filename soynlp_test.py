import collections
from funtiontable import FunctionTable
import functions as f
import pandas as pd
import nltk
import numpy as np
import scipy
import math
import re





feature_dict = collections.defaultdict(int)
pos_features = collections.defaultdict(int)
mpqa_features = collections.defaultdict(int)
vocab_features= collections.defaultdict(int)
cosine_features_dict= collections.defaultdict(int)

#기존의 파일 불러오기
#list_file = open('/Users/rjsgm/PycharmProjects/Covefefe_kor/output_folder/pos.txt', 'r',encoding='UTF-8').read().split('\n')

#새로운 pandas 형식의 txt pos,token 있는 txt파일 불러오기
list_file=pd.read_csv('/Users/rjsgm/PycharmProjects/Covefefe_kor/output_folder/pos.txt',names=['pos','token'])
sentence_file=pd.read_csv('/Users/rjsgm/PycharmProjects/Covefefe_kor/input_folder/a.txt',names=['sentence'])




# freq 가져오기
get_frequency_norms =True
get_pos_ratios=True
get_density=True
norms_freq =f.get_frequency_norms()
norms_word=list(norms_freq.keys())
norms_log10wf=list(norms_freq.values())



mpqa_list = f.get_mpqa_lexicon()
mpqa_words = mpqa_list[0]
mpqa_types = mpqa_list[1]
mpqa_polarities = mpqa_list[2]


total_tk=len(list_file.index)
pos_features['word_length'] =total_tk


inf_value=10^10
nan_value=-1



list_string_file=' '.join(list_file['pos'])
total_words=len(list_file['pos'])


def get_filler_counts(transcript_utterances_fillers):

    ''' This function extracts filler counts.
    Parameters:
    transcript_utterances_fillers: list of list of strings, transcript utterances with fillers included.
    Returns:
    filler_keys: list of strings, names of extracted features.
    filler_features: dictionary, mapping feature name to feature value.
    '''

    filler_keys = []
    filler_counts = {}

    regex_fillers = {'fillers': re.compile(r'^(?:(?:ah)|(?:eh)|(?:er)|(?:ew)|(?:hm)|(?:mm)|(?:uh)|(?:uhm)|(?:um))$'),
                     'um': re.compile(r'^(?:(?:uhm)|(?:um))$'),
                     'uh': re.compile(r'^(?:(?:ah)|(?:uh))$')}
    filler_keys = regex_fillers.keys()
    filler_counts = collections.defaultdict(int)

    if transcript_utterances_fillers is not None:
        for utt in transcript_utterances_fillers:
            for word in utt:
                for filler_type in filler_keys:
                    if regex_fillers[filler_type].findall(word):
                        filler_counts[filler_type] += 1

    return filler_keys, filler_counts


get_filler_counts(list_file)
print(get_filler_counts(list_file))