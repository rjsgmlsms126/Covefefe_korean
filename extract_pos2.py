import collections
from funtiontable import FunctionTable
import functions as f
import pandas as pd
import numpy as np
import nltk

import re

inf_value=-1


feature_dict = collections.defaultdict(int)
pos_features = collections.defaultdict(int)
mpqa_features = collections.defaultdict(int)

#기존의 파일 불러오기
#list_file = open('/Users/rjsgm/PycharmProjects/Covefefe_kor/output_folder/pos.txt', 'r',encoding='UTF-8').read().split('\n')

#새로운 pandas 형식의 txt pos,token 있는 txt파일 불러오기
list_file=pd.read_csv('/Users/rjsgm/PycharmProjects/Covefefe_kor/output_folder/pos.txt',names=['pos','token'])


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



nan_value=-1



list_string_file=' '.join(list_file['pos'])
total_words=len(list_file['pos'])

pos = list_file.loc[:, ['pos']]
token = list_file.loc[:, ['token']]
pos_list = pos.to_dict('list')
token_list = token.to_dict('list')

#print(token)
#print(token_list)






def get_cosine_distance(data):


    pos = data.loc[:, ['pos']]
    token = data.loc[:, ['token']]
    pos_list = pos.to_dict('list')
    token_list = token.to_dict('list')

    ''' This function extracts cosine distance features.
    Parameters:
    transcript_utterances: list of lists of strings (words), each row is a plaintext utterance in the transcript.
    stopwords: list of string, words to be removed.
    inf_value: int, value for infinity.
    Returns:
    cosine_keys: list of strings, names of extracted features.
    cosine_features_dict: dictionary, mapping feature name to feature value.
    '''

    cosine_keys = ["ave_cos_dist", "min_cos_dist", "cos_cutoff_00", "cos_cutoff_03", "cos_cutoff_05"]
    cosine_features_dict = {}



    for p_value in pos_list.values():
        pos_tokens=p_value
    for t_value in token_list.values():
        lemmatized_tokens=t_value
        for i in t_value:
            t_value[i]

    # REPETITION
    # Build a vocab for the transcript
    fdist_vocab = nltk.probability.FreqDist([t_value[i] for t_value in token_list.values() for i in t_value])
    vocab_words = list(fdist_vocab.keys())


    num_utterances = len(token_list.values())

    # Create a word vector for each utterance, N x V
    # where N is the num of utterances and V is the vocab size
    # The vector is 1 if the vocab word is present in the utterance,
    # 0 otherwise (i.e., one hot encoded).
    word_vectors = []
    for i,t_value in enumerate(token_list.values()):
        word_vectors.append(len(vocab_words)*[0]) # init
        for j in range(len(vocab_words)):
            if vocab_words[j] in t_value:
                word_vectors[i][j] += 1

    # Calculate cosine DISTANCE between each pair of utterances in
    # this transcript (many entries with small distances means the
    # subject is repeating a lot of words).
    average_dist = 0.0
    min_dist = 1.0
    num_similar_00 = 0.0
    num_similar_03 = 0.0
    num_similar_05 = 0.0
    num_pairs = 0
    for i in range(num_utterances):
        for j in range(i):
            # The norms of the vectors might be zero if the utterance contained only
            # stopwords which were removed above. Only compute cosine distance if the
            # norms are non-zero; ignore the rest.
            norm_i, norm_j = np.linalg.norm(word_vectors[i]), np.linalg.norm(word_vectors[j])
            if norm_i > 0 and norm_j > 0:
                cosine_dist = scipy.spatial.distance.cosine(word_vectors[i], word_vectors[j])
                if math.isnan(cosine_dist):
                    continue
                average_dist += cosine_dist
                num_pairs += 1
                if cosine_dist < min_dist:
                    min_dist = cosine_dist

                # Try different cutoffs for similarity
                if cosine_dist < 0.001: #similarity threshold
                    num_similar_00 += 1
                if cosine_dist <= 0.3: #similarity threshold
                    num_similar_03 += 1
                if cosine_dist <= 0.5: #similarity threshold
                    num_similar_05 += 1

    # The total number of unique utterance pairwise comparisons is <= N*(N-1)/2
    # (could be less if some utterances contain only stopwords and end up empty after
    # stopword removal).
    denom = num_pairs

    if denom >= 1:
        cosine_features = [average_dist * 1.0 / denom,
                           min_dist,
                           num_similar_00 * 1.0 / denom,
                           num_similar_03 * 1.0 / denom,
                           num_similar_05 * 1.0 / denom]
    else:
        # There are either no utterances or a single utterance -- no repetition occurs
        cosine_features = [inf_value, inf_value, 0, 0, 0]

    for ind_feat, feat_name in enumerate(cosine_keys):
        cosine_features_dict[feat_name] = cosine_features[ind_feat]

    return cosine_keys, cosine_features_dict

#a=get_vocab_richness_measures(list_file)
#print(a.vocab_keys)
#print(collections.Counter(vocab_features))
a=list(token_list.values())
print(len(a[0]))

#for i,t_values1 in enumerate(a[0]):
 #   print(t_values1)


