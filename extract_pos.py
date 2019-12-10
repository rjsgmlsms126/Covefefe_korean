from koalanlp.Util import initialize
from koalanlp.proc import Tagger
from koalanlp import API

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

initialize(EUNJEON='LATEST')

tagger = Tagger(API.EUNJEON)

filename = 'a.txt'
with open(filename,'r',encoding='UTF-8') as file_object:
    contents = file_object.read()


sentences = tagger(contents)

print("===== Sentence =====")
print(type(sentences))
f = open("pos.txt", 'w',encoding='UTF-8')
for i, sent in enumerate(sentences):
    print("===== Sentence #$i =====")
    sentenceswrite=sent.surfaceString()
    f.write(sentenceswrite)
    f.write("\n")

    print(sentenceswrite)
    print("# Analysis Result")
    # print(sent.singleLineString())

    for word in sent:
        wordId=word.getId()
        wordSurf=word.getSurface()
        wordSum=str(wordId) + wordSurf
       # f.write("Word ["+str(wordId)+"]"+" "+wordSurf+":"+"\n")
        print("Word [%s] %s = " % (word.getId(), word.getSurface()), end='')

        for morph in word:
            morphSurf=morph.getSurface()
            morphTag=morph.getTag()
            morphSum=str(morphSurf) + str(morphTag)+"\n"
            #f.write("\t"+str(morphSurf)+"/"+str(morphTag)+"\n")
            f.write(str(morphTag)+',')
            #f.write("%s/%s " % (morph.getSurface(), morph.getTag()), end='')
            print("%s/%s " % (morph.getSurface(), morph.getTag()), end='')

        print()
f.close()