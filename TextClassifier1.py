import jieba
import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

read_path = './data/IT'
file_list = []
for file_name in os.listdir(read_path):
    if file_name[0] == '.':
        pass
    else:
        file_list.append(file_name)

corpus = []
for file in file_list:
    seg_list = jieba.cut(open(read_path + '/' + file))
    result = []
    for seg in seg_list:
        if seg != ' ' and seg != "\n" and seg != '\t':
            result.append(seg)
    corpus.append(result)

vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfdif = transformer.fit_transform(vectorizer.fit_transform(corpus))
words = vectorizer.get_feature_names()
weight = tfdif.toarray()
