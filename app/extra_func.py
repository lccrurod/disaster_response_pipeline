# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 17:32:58 2021

@author: justC
"""
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def tokenize(text):
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stop_words:
            clean_tokens.append(clean_tok)

    return clean_tokens

class VerbCountExtractor(BaseEstimator, TransformerMixin):

    def verb_count(self, text):
        pos_tags = nltk.pos_tag(tokenize(text))
        return sum([1 for tag in pos_tags if tag[0:2] == 'VB'])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.verb_count)
        return pd.DataFrame(X_tagged)
    
if __name__ == '__main__':
    print('loaded modules')