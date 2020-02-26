#! usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
import codecs
import pandas as pd
import numpy as np
from feature_extractor import FeatureExtractor
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.externals import joblib
import sklearn.metrics
from evaluator import evaluate
import csv
import tarfile
import os

# read embeddings and put them in an hashmap
embeddings = {}

with open('glove.twitter.27B.50d.txt') as glove:
    for line in glove:
        line = line.strip().split(" ")
        embeddings[str(line[0])] = [float(i) for i in line[1:]]


class Token:

    def __init__(self, word, lemma, pos, cpos):
        self.word = word
        self.lemma = lemma
        self.cpos = cpos
        self.pos = pos


class Sentence:

    def __init__(self):
        self.tokens = []

    def add_token(self, token):
        self.tokens.append(token)


class Document(object):

    def __init__(self):
        self.sentences = []

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def get_tokens(self):
        return [token for sentence in self.sentences for token in sentence.tokens]


class InputReader(object):

    def __init__(self, input_file_name):
        self.current_sentence = Sentence()
        self.input_file_name = input_file_name

    def generate_documents(self):
        current_document = Document()
        current_sentence = Sentence()
        self.input_file = codecs.open(self.input_file_name, 'r', 'utf-8')
        while True:
            l = self.input_file.readline()
            if l == '\n':
                current_document.add_sentence(current_sentence)
                current_sentence = Sentence()
            elif "newdoc" in l:
                if current_document.sentences:
                    yield current_document
                current_document = Document()
                # Read ID
                if "id" in l:
                    current_document.id = re.compile('(\d+)').findall(l)[0]
                else:
                    current_document.id = "0"

                # Read labels
                # label neutro se 00, positivo 10, posneg 11, - non gestisce n label, è sempre una
                # TODO cambiare la funzione e decidere come assegnare label
                # TODO n modelli - leggere le label con un flag per capire quale label leggere, oppure leggere colonna diversa
                # passare come feature la label per vedere se è corretto il classificatore
                is_positive, is_negative = False, False
                if 'emo=1' in l:
                    is_positive = True
                if 'emo=0' in l:
                    is_negative = True
                if is_positive:
                    current_document.label = "POS"
                elif is_negative:
                    current_document.label = "NEG"
                else:
                    current_document.label = "O"
            elif '#' in l and 'newdoc' not in l:
                pass
            elif l == '':
                current_document.add_sentence(current_sentence)
                yield current_document
                raise StopIteration
            else:
                split_token = l.rstrip('\n').split("\t")
                tok = Token(split_token[1], split_token[2],
                            split_token[3], split_token[4])
                current_sentence.add_token(tok)


class ToySentimentClassifier(object):

    def __init__(self):
        self.feature_extractor = FeatureExtractor()

    def extract_features(self, doc):
        all_features = {}
        for i in range(1, 4):  # range numero di ngrammi da calcolare
            all_features.update(
                self.feature_extractor.extract_word_ngrams(doc, i))  # restituisce features degli ngrammi
        for i in range(1, 4):
            all_features.update(self.feature_extractor.extract_lemma_ngrams(doc, i))
        for i in range(3, 6):
            all_features.update(self.feature_extractor.compute_n_chars(doc, i))
        all_features.update(self.feature_extractor.compute_document_length(doc))
        all_features.update(self.feature_extractor.compute_embeddings(doc, embeddings))
        return all_features

    def train(self, model_name, input_file_name):
        reader = InputReader(input_file_name)
        all_docs = []
        for doc in reader.generate_documents():
            doc.features = self.extract_features(doc)
            all_docs.append(doc)  # lista con documenti+features del documento

        # Encoding of samples
        all_collected_feats = [doc.features for doc in all_docs]
        # trasformare features in vettori (vettore singolo doc, matrice collezione di documenti)
        X_dict_vectorizer = DictVectorizer(
            sparse=True)  # funzione di sklearn - prende un dizionario key:id feature value: valore feature
        # trasforma in matrice
        encoded_features = X_dict_vectorizer.fit_transform(
            all_collected_feats)  # crea matrice (sparse=True matrice con hashmap)

        # Scale to increase performances and reduce training time
        # vogliamo che ogni feature contribuisca in modo equo - scalare feature fra 0 e 1
        scaler = preprocessing.StandardScaler(with_mean=False).fit(
            encoded_features)  # calcola i parametri che devono essere usati per scalare
        encoded_scaled_features = scaler.transform(encoded_features)  # scala i parametri

        # Encoding of labels (Y)
        label_encoder = preprocessing.LabelEncoder()  # è multilabel, anche le classi vanno scalate
        label_encoder.fit([doc.label for doc in all_docs])
        encoded_labels = label_encoder.transform([doc.label for doc in all_docs])

        # Classifier Algorithm
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        clf = SVC(kernel='linear', C=1e3)
        # Cross validation
        cross_val_scores = cross_validate(clf, encoded_scaled_features, encoded_labels, cv=10, scoring=scoring)

        print('accuracy\tprecision\trecall\tf1\n')
        print(str(np.average(cross_val_scores['test_accuracy']))
              + '\t' + str(np.average(cross_val_scores['test_precision']))
              + '\t' + str(np.average(cross_val_scores['test_recall']))
              + '\t' + str(np.average(cross_val_scores['test_f1'])) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sentiment Classifier')
    parser.add_argument('-i', '--input_file', help='The input file in CONLL Format', required=True)
    parser.add_argument('-m', '--model_name', help='The model name', required=True)
    parser.add_argument('-o', '--output_file', help='The output file')
    parser.add_argument('-t', '--train', help='Trains the model', action='store_true')
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    model_name = args.model_name
    train_mode = args.train
    train_mode = True
    classifier = ToySentimentClassifier()
    if train_mode:
        classifier.train(model_name, input_file)

