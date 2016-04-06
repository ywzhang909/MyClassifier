# encoding:utf-8

import os
import sys
import jieba
import string
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


def get_file_list(argv: object) -> object:
    read_path = argv[1]
    file_list = []
    read_files = os.listdir(read_path)
    for f in read_files:
        if f[0] == '.':
            pass
        else:
            file_list.append(f)
    return file_list, read_path


def partition(argv, fpath):
    save_path = './segfile'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    filename = argv
    f = open(fpath + filename, 'r')
    file_list = f.read()
    f.close()

    seg_list = jieba.cut(file_list, cut_all=True)

    # handle the symbols in the segments
    result = []
    for seg in seg_list:
        seg = ''.join(seg.split())
        if seg != '' and seg != '\r\n':
            result.append(seg)

    f = open(save_path + '/' + filename + '-seg.txt', 'w+')
    f.write(' '.join(result))
    f.close()


def tf_idf(seg_files):
    seg_path = './segfile/'
    corpus = []
    for file in seg_files:
        fname = seg_path + file
        f = open(fname, 'r+')
        content = f.read()
        f.close()
        corpus.append(content)

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfdif = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names()
    weight = tfdif.toarray()

    save_path = './tfidffile'
    if not os._exists(save_path):
        os.mkdir(save_path)

    for i in range(len(weight)):
        print('--------Writing all the tf-idf in the', i, u' file into ', save_path + '/' + string.zfill(i, 5) + '.txt',
              '--------')
        f = open(save_path + '/' + string.zfill(i, 5) + '.txt', 'w+')
        for j in range(len(word)):
            f.write(word[j] + ' ' + str(weight[i][j]) + '\r\n')
        f.close()


if __name__ == '__main':
    (all_file, path) = get_file_list(sys.argv)
    for files in all_file:
        print('Parting...')
        partition(files, path)
    tf_idf(all_file)
