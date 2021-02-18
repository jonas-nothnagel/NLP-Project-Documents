# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:51:22 2020

@author: jonas.nothnagel

@title: K-means clustering with tf-idf embeddings. 
"""
import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import seaborn as sns

import matplotlib.pyplot as plt

#Sklearn for feature extraction
def feature_extraction(text, min_ngram, max_ngram, min_df_value, max_df_value):
    
    clean_text = text
    tf_idf_vectorizor = TfidfVectorizer(ngram_range = (min_ngram,max_ngram), min_df = min_df_value, max_df = max_df_value)

    vec = tf_idf_vectorizor.fit(clean_text)
    tf_idf = vec.transform(clean_text)
    tf_idf_norm = normalize(tf_idf)
    tf_idf_array = tf_idf_norm.toarray()
    #look at features
    pd.DataFrame(tf_idf_array, columns=tf_idf_vectorizor.get_feature_names()).head()
    
    return tf_idf_array, tf_idf_vectorizor, vec


#PCA for dimension reduction:
def pca_d_reduction(vector, components):
    sklearn_pca = PCA(n_components = components)
    pca = sklearn_pca.fit(vector)
    print('number of components:',pca.n_components_)
    Y_sklearn = sklearn_pca.fit_transform(vector)
    
    return Y_sklearn, pca


def get_top_features_cluster(tf_idf_array, prediction, n_feats, vetorizer):
    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction==label) # indices for each cluster
        x_means = np.mean(tf_idf_array[id_temp], axis = 0) # returns average score across cluster
        sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores
        features = vetorizer.get_feature_names()
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns = ['features', 'score'])
        dfs.append(df)
    return dfs

def infer_query(query, vec, pca, kmeans, dfs, dictlist, dataframe):
    
    query = [query]
    query_vec = vec.transform(query)
    query_vec_norm = normalize(query_vec)
    query_vec_array = query_vec_norm.toarray()
    query_vec_pca = pca.transform(query_vec_array)

    infer = kmeans.predict(query_vec_pca)
    number = int(infer)
    plt.figure(figsize=(6,4))
    sns.barplot(x = 'score' , y = 'features', orient = 'h' , data = dfs[number][:7])

    projects = dictlist[number]
    project_id = []
    for i in projects[1]:
        project_id.append(dataframe.country.iloc[i])

    print('Query:', query)
    print('')
    print('Countries that are part of this cluster:')
    print(project_id)
    print('')
    print('top words of predicted cluster:')