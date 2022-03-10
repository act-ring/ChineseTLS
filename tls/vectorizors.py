#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Li Xi
@email: whxf_lixi@buaa.edu.cn
@time: 2021/7/29 21:06
"""

import gensim
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from . import data
import numpy as np
from sentence_transformers import SentenceTransformer


class Vectorizer:

    def get_vector(self, input: List[str]):
        raise NotImplementedError


class TfIdfVectorizer(Vectorizer):
    def __init__(self, dataset_path, language="chinese"):
        """
        :param language: ["chinese", "english"]
        """
        if language == "english":
            self.model = TfidfVectorizer(lowercase=True, stop_words='english')
        else:
            stopwords = data.load_stopwords('./extra_data/cn_stopwords.txt')
            self.model = TfidfVectorizer(lowercase=True, stop_words=stopwords)

        dataset = data.Dataset(dataset_path)
        articles = []
        for i, collection in enumerate(dataset.collections):
            articles += list(collection.articles())
        self.model.fit([s.token for a in articles for s in a.sentences])

    def get_vector(self, input: List[str]):
        return self.model.transform([" ".join(input)]).toarray()[0]

    def get_matrix(self, inputs):
        return self.model.transform(inputs)


class Doc2vecVectorizer(Vectorizer):
    def __init__(self, language="chinese"):
        """
        :param language: ["chinese", "english"]
        """
        self.model = gensim.models.Doc2Vec.load('/home/LAB/lixi/projects/adaptive-tls/vectorizers/doc2vec.model')

    def get_vector(self, input: List[str]):
        return self.model.infer_vector(input)

    def get_matrix(self, inputs):
        return np.array([self.get_vector(list(s.split())) for s in inputs])


class SentBertVectorizer(Vectorizer):
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    def get_vector(self, input: List[str]):
        return self.model.encode("".join(input))

    def get_matrix(self, inputs):
        return np.array([self.model.encode("".join(s.split())) for s in inputs])

# class Word2vecVectorizer(Vectorizer):
