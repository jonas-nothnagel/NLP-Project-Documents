# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 15:08:03 2021

@author: jonas.nothnagel

@title: prediction functions
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))
sys.setrecursionlimit(20500)
import pandas as pd

'''Plotting'''
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

'''helper'''
import clean_dataset as clean

'''features'''
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import label_binarize

'''Classifiers'''
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB


'''Metrics/Evaluation'''
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from scipy import interp
from itertools import cycle
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

import operator    
import joblib


#%%
hat = []
hat_2 = []
def predict_text(input_string, category, d, spacy = False, basic_clean = True, parent_only = True, tfidf = True):
    

    print('______')
    input_list = [input_string]
    input_df = pd.DataFrame(input_list, columns =['input_text'])
    
    """input text will be normalise to the standard of training data:"""
    if basic_clean == True:       
        input_df['input_text'] = input_df['input_text'].apply(clean.basic)
        clean_df = pd.Series(input_df['input_text'])
    else:
        pass
    if spacy == True:
        input_df['input_text'] = input_df['input_text'].apply(clean.spacy_clean)
        clean_df = pd.Series(input_df['input_text'])
    else:
        pass

    
    for key, value in d.items():
        
        if len(key) > 50:
            key = key[0:20]            
        else:
            pass
            
        """input text will be vectorised/embedded:

            tf-idf
        """


        """load vectorizer and LSA dimension reducer"""

        tfidf_vectorizer = joblib.load('../../models/tf_idf/'+category+'/'+key+'_'+'vectorizer.sav')        
        lsa = joblib.load('../../models/tf_idf/'+category+'/'+key+'_'+'lsa.sav')

        vector_df = tfidf_vectorizer.transform(clean_df)
        vector_df = lsa.transform(vector_df)

        """load models:"""
        clf = joblib.load('../../models/tf_idf/'+category+'/'+key+'_'+'model.sav')

        """predict"""
        y_hat_parent = clf.predict(vector_df)
        y_prob_parent = clf.predict_proba(vector_df)

        if y_hat_parent == 1:
            print(key)
            print("YES. Confidence:", y_prob_parent[0][1].round(2)*100, "%")
            hat.append(y_prob_parent[0][1].round(2)*100)
            
            if parent_only == False: 
                """predicting sub categories""" 
                print('____________________________________________') 
                print("predicting sub categories....")
                for sub_key in value:

                    if len(sub_key) > 50:
                        sub_key = sub_key[0:20]  
                    else:
                        pass

                    tfidf_vectorizer = joblib.load('../../models/tf_idf/'+category+'/'+sub_key+'_'+'vectorizer.sav')        
                    lsa = joblib.load('../../models/tf_idf/'+category+'/'+sub_key+'_'+'lsa.sav')

                    vector_df = tfidf_vectorizer.transform(clean_df)
                    vector_df = lsa.transform(vector_df)

                    """load models:"""
                    clf = joblib.load('../../models/tf_idf/'+category+'/'+sub_key+'_'+'model.sav')

                    """predict"""
                    y_hat = clf.predict(vector_df)
                    y_prob = clf.predict_proba(vector_df)

                    if y_hat == 1:
                        print(sub_key)
                        print("YES. Confidence:", y_prob[0][1].round(2)*100, "%")
                        hat_2.append(y_prob[0][1].round(2)*100)

                    if y_hat == 0:
                        print(sub_key)
                        print("NO Confidence:", y_prob[0][0].round(2)*100, "%")
                        hat.append(y_prob[0][0].round(2)*100)            

                    print('')
                print('____________________________________________')      
                
            else:
                pass
            
        if y_hat_parent == 0:
            print(key)
            print("NO Confidence:", y_prob_parent[0][0].round(2)*100, "%")
            hat.append(y_prob_parent[0][0].round(2)*100)
        
        print('______')
        
    return y_hat_parent, y_prob_parent, hat, hat_2


hat = []
hat_2 = []
def predict_text_all(input_string, category, d, spacy = False, basic_clean = True, parent_only = True, tfidf = True):
    
    '''import data'''
    df = pd.read_csv(os.path.abspath(os.path.join('..', 'data/processed/'))+'/taxonomy_final.csv')  

    cats = df.drop(columns=['PIMS_ID','all_text_clean', 'all_text_clean_spacy'])
    categories = cats.columns.tolist()

    print('______')
    input_list = [input_string]
    input_df = pd.DataFrame(input_list, columns =['input_text'])
    
    """input text will be normalise to the standard of training data:"""
    if basic_clean == True:       
        input_df['input_text'] = input_df['input_text'].apply(clean.basic)
        clean_df = pd.Series(input_df['input_text'])
    else:
        pass
    if spacy == True:
        input_df['input_text'] = input_df['input_text'].apply(clean.spacy_clean)
        clean_df = pd.Series(input_df['input_text'])
    else:
        pass

    
    for key, value in d.items():
        
        if df[key].sum(axis=0) > 20:
            print(df[key].sum(axis=0))
            if len(key) > 50:
                key = key[0:20]            
            else:
                pass

            """input text will be vectorised/embedded:

                tf-idf
            """


            """load vectorizer and LSA dimension reducer"""

            tfidf_vectorizer = joblib.load('../models/tf_idf/'+category+'/'+key+'_'+'vectorizer.sav')        
            lsa = joblib.load('../models/tf_idf/'+category+'/'+key+'_'+'lsa.sav')

            vector_df = tfidf_vectorizer.transform(clean_df)
            vector_df = lsa.transform(vector_df)

            """load models:"""
            clf = joblib.load('../models/tf_idf/'+category+'/'+key+'_'+'model.sav')

            """predict"""
            y_hat_parent = clf.predict(vector_df)
            y_prob_parent = clf.predict_proba(vector_df)

            if y_hat_parent == 1:
                print(key)
                print("YES. Confidence:", y_prob_parent[0][1].round(2)*100, "%")
                hat.append(y_prob_parent[0][1].round(2)*100)

                if parent_only == False: 
                    """predicting sub categories""" 
                    print('____________________________________________') 
                    print("predicting sub categories....")
                    for sub_key in value:
                        if df[sub_key].sum(axis=0) > 20:
                            print(df[sub_key].sum(axis=0))
                            if len(sub_key) > 50:
                                sub_key = sub_key[0:20]  
                            else:
                                pass

                            tfidf_vectorizer = joblib.load('../models/tf_idf/'+category+'/'+sub_key+'_'+'vectorizer.sav')        
                            lsa = joblib.load('../models/tf_idf/'+category+'/'+sub_key+'_'+'lsa.sav')

                            vector_df = tfidf_vectorizer.transform(clean_df)
                            vector_df = lsa.transform(vector_df)

                            """load models:"""
                            clf = joblib.load('../models/tf_idf/'+category+'/'+sub_key+'_'+'model.sav')

                            """predict"""
                            y_hat = clf.predict(vector_df)
                            y_prob = clf.predict_proba(vector_df)

                            if y_hat == 1:
                                print(sub_key)
                                print("YES. Confidence:", y_prob[0][1].round(2)*100, "%")
                                hat_2.append(y_prob[0][1].round(2)*100)

                            if y_hat == 0:
                                print(sub_key)
                                print("NO Confidence:", y_prob[0][0].round(2)*100, "%")
                                hat.append(y_prob[0][0].round(2)*100)            

                            print('')
                        print('____________________________________________')      

                else:
                    pass

            if y_hat_parent == 0:
                print(key)
                print("NO Confidence:", y_prob_parent[0][0].round(2)*100, "%")
                hat.append(y_prob_parent[0][0].round(2)*100)

            print('______')
        
    return y_hat_parent, y_prob_parent, hat, hat_2
