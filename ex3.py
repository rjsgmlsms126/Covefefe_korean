import re
import csv
import pandas as pd
#df=pd.read_csv('C:/Users/rjsgm/PycharmProjects/Covefefe_kor/dic/mpqa_kor_1.csv', "r",encoding='utf-8',names=['sub','word','num','pol'],delimiter=',')

#print(df.loc[3])



#with open('C:/Users/rjsgm/PycharmProjects/Covefefe_kor/dic/mpqa_kor_1.csv', "r",encoding='utf-8') as f:
    #lines = f.readlines()
    #print(lines[1])
  #  words = [re.match('word1=(.*)', line.split()[2]).groups()[0] for line in lines]
  #  types = [re.match('type=(.*)subj', line.split()[0]).groups()[0] for line in lines]
   # polarities = [re.search('polarity=(.*)', line).groups()[0] for line in lines]


   # reader=csv.reader(f)
   # a_list=list(reader)

   # word=[word[1] for word in a_list]

def get_mpqa_lexicon():

    with open('C:/Users/rjsgm/PycharmProjects/Covefefe_kor/dic/mpqa_kor_1.csv', "r",encoding='utf-8') as f:
        reader = csv.reader(f)
        mpqa_list = list(reader)

    words =[word[1] for word in mpqa_list]
    types = [type[0] for type in mpqa_list]
    polarities = [polar[3] for polar in mpqa_list]

    return [words, types, polarities]
print(get_mpqa_lexicon())