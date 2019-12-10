filename = 'output.txt'
with open(filename,'r',encoding='UTF-8') as file_object:
    contents = file_object.read()
text = contents.strip()




def get_pos_feature():
    pos_keys = ['word_length', 'NID']
    if get_pos_counts:
        pos_keys += ['nouns', 'verbs', 'inflected_verbs', 'light', 'function', 'pronouns', 'determiners', 'adverbs',
                     'adjectives', 'prepositions',
                     'coordinate', 'subordinate', 'demonstratives']
    if get_pos_ratios:
        pos_keys += ['nvratio', 'prp_ratio', 'noun_ratio', 'sub_coord_ratio']
    if get_frequency_norms:
        pos_keys += ['frequency', 'noun_frequency', 'verb_frequency']
    if get_image_norms:
        pos_keys += ['aoa', 'imageability', 'familiarity',
                     'noun_aoa', 'noun_imageability', 'noun_familiarity',
                     'verb_aoa', 'verb_imageability', 'verb_familiarity']
    if get_anew_norms:
        pos_keys += ['noun_anew_val_mean', 'noun_anew_val_std', 'noun_anew_aro_mean', 'noun_anew_aro_std',
                     'noun_anew_dom_mean', 'noun_anew_dom_std',
                     'verb_anew_val_mean', 'verb_anew_val_std', 'verb_anew_aro_mean', 'verb_anew_aro_std',
                     'verb_anew_dom_mean', 'verb_anew_dom_std',
                     'anew_val_mean', 'anew_val_std', 'anew_aro_mean', 'anew_aro_std', 'anew_dom_mean', 'anew_dom_std']
    if get_warringer_norms:
        pos_keys += ['warr_val_mean', 'warr_val_std', 'warr_val_rat', 'warr_aro_mean', 'warr_aro_std',
                     'warr_aro_rat', 'warr_dom_mean', 'warr_dom_std', 'warr_dom_rat',
                     'warr_val_mean_nn', 'warr_val_std_nn', 'warr_val_rat_nn', 'warr_aro_mean_nn', 'warr_aro_std_nn',
                     'warr_aro_rat_nn', 'warr_dom_mean_nn', 'warr_dom_std_nn', 'warr_dom_rat_nn',
                     'warr_val_mean_vb', 'warr_val_std_vb', 'warr_val_rat_vb', 'warr_aro_mean_vb', 'warr_aro_std_vb',
                     'warr_aro_rat_vb', 'warr_dom_mean_vb', 'warr_dom_std_vb', 'warr_dom_rat_vb']
    if get_density:
        pos_keys += ['prop_density', 'content_density']

    hangul = re.compile('[^ \u3131-\u3163\uac00-\ud7a3]+')
