import os
import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

seg_path = "./segfiles"

def get_files_list(path):
    file_list = []
    read_files = os.listdir(path)
    for f in read_files:
        if f[0] == '.':
            pass
        else:
            file_list.append(f)
    return file_list


def partition(file_name, path):
    if not os.path.exists(seg_path ):
        os.mkdir(seg_path)
    content = open(path + '/' + file_name).read()
    seg_list = jieba.cut(content)
    result = []
    for seg in seg_list:
        if seg != ' ' and seg != "\n" and seg != '\t':
            result.append(seg)
    f = open(seg_path + '/' + file_name + '-seg.txt', 'w+', encoding='GB2312')
    f.write(' '.join(result))
    f.close()


def dfidf(seg_files):
    corpus = []
    for file in seg_files:
        file_path = seg_path + file
        content = open(file_path).read()
        corpus.append(content)
    vetorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfdif = transformer.fit_transform(vetorizer.fit_transform(corpus))
    word = vetorizer.get_feature_names()
    weight = tfdif.toArray()



if __name__ == '__main__':
    file_path = "./data/IT"
    for file in get_files_list(file_path):
        partition(file, file_path)
