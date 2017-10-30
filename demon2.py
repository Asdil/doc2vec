#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# 另外一种输入，所有数据在一个txt文件里， 一句话相当于一个文本
import os
from gensim.models.doc2vec import TaggedLineDocument, Doc2Vec
if __name__ == '__main__':
    train_corpus = os.getcwd() + '/data/demon2_train.txt'
    docments = TaggedLineDocument(train_corpus)
    model = Doc2Vec(docments, size=20, window=10, min_count=2)
    model.save(os.getcwd() + '/model/model2')  # 保存模型
