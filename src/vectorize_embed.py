#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 17:31:26 2020

@author: jonas & oumaima

@title: vectorizations and embeddings

@description: scripts to transform text into features
"""
#%%

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

#%%

def get_embeddings(model_name, sentences):
    
     model = SentenceTransformer(model_name)
    
     for i in range(len(sentences)):
         sentences[i] = str(sentences[i])
    
     sentence_embeddings = model.encode(sentences)
    
     return sentence_embeddings


def tf_idf(text, min_ngram = 1, max_ngram = 2, min_df_value=0.1, max_df_value=0.9):
    
    """
    takes in text and returns tf-idf vectors. 
    Let's user specifiy parameters and uses sklearns default settings otherwise.
    """
    clean_text = text
    tf_idf_vectorizor = TfidfVectorizer(ngram_range = (min_ngram,max_ngram), min_df = min_df_value, max_df = max_df_value)

    vec = tf_idf_vectorizor.fit(clean_text)
    tf_idf = vec.transform(clean_text)
    tf_idf_norm = normalize(tf_idf)
    tf_idf_array = tf_idf_norm.toarray()
    return tf_idf_array, tf_idf_vectorizor, vec

def tf_idf_oumaima(documents):
    
    """
    tf_idf vectorizer
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    
    return X

def elmo():
    
    """
    elmo embeddings:
    """
    #get elmo from tensorflow hub



