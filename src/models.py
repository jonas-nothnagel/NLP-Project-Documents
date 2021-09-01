# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:15:47 2020

@author: jonas.nothnagel

@title: models for classification
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
#Set up models

#Function to compare and store best performing models from a defined score:
def model_score_df(model_dict, category, taxonomy_label, X_train, X_test, y_train, y_test):   
    

    models, model_name, ac_score_list, p_score_list, r_score_list, f1_score_list = [], [], [], [], [], []
    
    for k,v in model_dict.items():   

        
        v.fit(X_train, y_train)
        
        model_name.append(k)
        models.append(v)
        
        y_pred = v.predict(X_test)
#         ac_score_list.append(accuracy_score(y_test, y_pred))
#         p_score_list.append(precision_score(y_test, y_pred, average='macro'))
#         r_score_list.append(recall_score(y_test, y_pred, average='macro'))
        f1_score_list.append(f1_score(y_test, y_pred, average='macro'))
#         model_comparison_df = pd.DataFrame([model_name, ac_score_list, p_score_list, r_score_list, f1_score_list]).T
#         model_comparison_df.columns = ['model_name', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']
#         model_comparison_df = model_comparison_df.sort_values(by='f1_score', ascending=False)

    results = dict(zip(models, f1_score_list))
    name = dict(zip(model_name, f1_score_list))    
    #return best performing model according to f1_score
    best_clf = max(results.items(), key=operator.itemgetter(1))[0]
    best_name = max(name.items(), key=operator.itemgetter(1))[0]
    
    if len(category) > 50:
        shorter = category[:20]
        print(shorter)
            
        #save best performing model
        filename = '../../models/tf_idf/'+taxonomy_label+'/'+shorter+'_'+best_name+'_'+'model.sav'
        joblib.dump(best_clf, filename)

        #save best performing model without name appendix
        filename = '../../models/tf_idf/'+taxonomy_label+'/'+shorter+'_'+'model.sav'
        joblib.dump(best_clf, filename)  
    
    else:

        #save best performing model
        filename = '../../models/tf_idf/'+taxonomy_label+'/'+category+'_'+best_name+'_'+'model.sav'
        joblib.dump(best_clf, filename)

        #save best performing model without name appendix
        filename = '../../models/tf_idf/'+taxonomy_label+'/'+category+'_'+'model.sav'
        joblib.dump(best_clf, filename)      
    
    return results

def model_score_df_all(model_dict, category, taxonomy_label, X_train, X_test, y_train, y_test):   
    

    models, model_name, ac_score_list, p_score_list, r_score_list, f1_score_list = [], [], [], [], [], []
    
    for k,v in model_dict.items():   

        
        v.fit(X_train, y_train)
        
        model_name.append(k)
        models.append(v)
        
        y_pred = v.predict(X_test)
#         ac_score_list.append(accuracy_score(y_test, y_pred))
#         p_score_list.append(precision_score(y_test, y_pred, average='macro'))
#         r_score_list.append(recall_score(y_test, y_pred, average='macro'))
        f1_score_list.append(f1_score(y_test, y_pred, average='macro'))
#         model_comparison_df = pd.DataFrame([model_name, ac_score_list, p_score_list, r_score_list, f1_score_list]).T
#         model_comparison_df.columns = ['model_name', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']
#         model_comparison_df = model_comparison_df.sort_values(by='f1_score', ascending=False)

    results = dict(zip(models, f1_score_list))
    name = dict(zip(model_name, f1_score_list))    
    #return best performing model according to f1_score
    best_clf = max(results.items(), key=operator.itemgetter(1))[0]
    best_f1 = max(results.items(), key=operator.itemgetter(1))[1]
    best_name = max(name.items(), key=operator.itemgetter(1))[0]
    
    if len(category) > 50:
        shorter = category[:20]
        print(shorter)
            
        #save best performing model
        filename = '../models/tf_idf/'+taxonomy_label+'/'+shorter+'_'+best_name+'_'+'model.sav'
        joblib.dump(best_clf, filename)

        #save best performing model without name appendix
        filename = '../models/tf_idf/'+taxonomy_label+'/'+shorter+'_'+'model.sav'
        joblib.dump(best_clf, filename)  
    
    else:

        #save best performing model
        filename = '../models/tf_idf/'+taxonomy_label+'/'+category+'_'+best_name+'_'+'model.sav'
        joblib.dump(best_clf, filename)

        #save best performing model without name appendix
        filename = '../models/tf_idf/'+taxonomy_label+'/'+category+'_'+'model.sav'
        joblib.dump(best_clf, filename)      
    
    return results, best_f1

# run logistic regression classification and display classification report:
def binary_log_classifier(dataframe, text, category, weighted = False):
    X_train, X_test, y_train, y_test = train_test_split(dataframe[text],
                                                        dataframe[category].values,
                                                        test_size = .3,
                                                        random_state = 1,
                                                        shuffle = True)
    print('training size:', len(X_train))
    print('test size:', len(X_test))
    print('distribution of tagged projects:', dataframe[category].value_counts())
    #if weighted == True:
        #class_weights = tools.get_class_weights(y_train)
        #print(class_weights)
    #else: 
        #class_weights = None

    '''extract features using tfidf vecorization:'''
    vectorizer = TfidfVectorizer(ngram_range = (1,2), min_df = 0.01, max_df = 0.95)
    vect = vectorizer.fit(X_train)
    X_train = vect.transform(X_train)
    X_test = vect.transform(X_test)

    pipe = Pipeline([('classifier' , LogisticRegression())])

    # Create param grid.
    param_grid = [
        {'classifier' : [LogisticRegression(class_weight = "balanced")],
         'classifier__penalty' : ['l1', 'l2'],
        'classifier__C' : np.logspace(-4, 4, 20),
        'classifier__solver' : ['liblinear ','lbfgs']}
    ]

    # Create grid search object
    clf = GridSearchCV(pipe, param_grid = param_grid, cv = 15, verbose=True, n_jobs=-1, scoring = 'f1')

    # Fit on data
    best_clf = clf.fit(X_train, y_train)
    print('')
    print('Training accuracy:', best_clf.score(X_train, y_train).round(3))
    print('Test accuracy:', best_clf.score(X_test, y_test).round(3))
    y_hat = best_clf.predict(X_test)
    print('recall', recall_score(y_test, y_hat))
    print('f1_score', f1_score(y_test, y_hat))
    
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_hat))    
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
    specificity = tn / (tn+fp)
    print('specificity is:', specificity)

    return best_clf, vectorizer, y_train


