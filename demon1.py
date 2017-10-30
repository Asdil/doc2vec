#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
from nltk.tokenize import WordPunctTokenizer
from gensim.models.doc2vec import TaggedDocument, Doc2Vec


# 读取/data下的文件
def read_file(path):
    raw_documents = []
    for root, dirs, files in path:
        for name in files:
            f = open(os.path.join(root, name), 'r')
            raw = f.read()
            raw_documents.append(raw)
    return raw_documents


# 创建输入数据
def create_input(raw_documents, stop_word):
    corpora_documents = []
    for i, documents in enumerate(raw_documents):
        documents = WordPunctTokenizer().tokenize(documents)  # 分词
        documents = [word for word in documents if word not in stop_word]  # 去停顿词
        documents = TaggedDocument(words=documents, tags=[i])  # 组成输入数据格式
        corpora_documents.append(documents)
    return corpora_documents


# 训练模型
def train_model(corpora_documents):
    model = Doc2Vec(size=40, min_count=2, iter=20)
    model.build_vocab(corpora_documents)
    model.train(corpora_documents)
    return model

if __name__ == '__main__':

    # 训练模型
    train_path = os.walk(os.path.realpath(os.getcwd() + '/data/train'))   # 文本路径
    stop_word = [line.strip() for line in open('stopword.txt').readlines()]  # 加载停顿词
    raw_documents = read_file(train_path)  # 读取txt文件
    corpora_documents = create_input(raw_documents, stop_word)  # 转换为Doc2Vec输入文件
    model = train_model(corpora_documents)
    model.save(os.getcwd() + '/model/model1') # 保存模型

    # 测试数据
    output_file = "data/vectors_result.txt"
    test_path = os.walk(os.path.realpath(os.getcwd() + '/data/test'))   # 文本路径
    raw_documents = read_file(test_path)  # 读取txt文件
    output = open(output_file, "w")
    for document in raw_documents:
        inferred_vector = model.infer_vector(document)
        output.write(" ".join(str(num) for num in inferred_vector) + '\n')
    output.flush()
    output.close()










