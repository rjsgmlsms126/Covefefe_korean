from koalanlp.Util import initialize
from koalanlp.proc import Tagger
from koalanlp import API
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

initialize(EUNJEON='LATEST')

tagger = Tagger(API.EUNJEON)

filename = 'a.txt'
with open(filename,'r',encoding='UTF-8') as file_object:
    contents = file_object.read()


sentences = tagger(contents)

#print("===== Sentence =====")

f = open("token.txt", 'w',encoding='UTF-8')
for i, sent in enumerate(sentences):
    #print("===== Sentence #$i =====")
    sentenceswrite=sent.surfaceString()
    #f.write(sentenceswrite)

    #print(sentenceswrite)
    #print("# Analysis Result")
    # print(sent.singleLineString())
    for word in sent:

        wordId=word.getId()
        wordSurf=word.getSurface()
        wordSum=str(wordId) + wordSurf

        f.write(wordSurf+',')
        #print(word.getSurface()+',')

        for morph in word:
            morphSurf=morph.getSurface()
            morphTag=morph.getTag()
            morphSum=str(morphSurf) + str(morphTag)+"\n"
            #f.write("\t"+str(morphSurf)+"/"+str(morphTag)+"\n")
            #f.write(str(morphTag)+',')
            #f.write("%s/%s " % (morph.getSurface(), morph.getTag()), end='')
            #print("%s/%s " % (morph.getSurface(), morph.getTag()), end='')

       # print()
f.close()

filename = 'token.txt'
with open(filename,'r',encoding='UTF-8') as file_object:
    contents = file_object.read()

filename1='stopwords.txt'
with open(filename1,'r',encoding='UTF-8') as file_object:
    contents1 = file_object.read()

f = open("token1.txt", 'w',encoding='UTF-8')

stopwords=contents1.split('\n')
word_tokens=word_tokenize(contents)

result =""
for w in word_tokens:
    if w not in stopwords:
        result=result+w+' '
print(type(word_tokens))
print(type(result))

def clean_str(text):
    pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)' # E-mail제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+' # URL제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '<[^>]*>'         # HTML 태그 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '[^\w\s]'         # 특수기호제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern='[-‘·=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]'
    text=re.sub(pattern=pattern,repl='',string=text)
    pattern ='[\d,]*'
    text=re.sub(pattern=pattern,repl='',string=text)
    return text


clean_str(result)
print(result)
f.write(result)