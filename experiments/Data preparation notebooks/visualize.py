#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:06:05 2020

@author: jonas & oumaima

@title: visualisations
"""
import holoviews as hv
from holoviews import opts, dim
import holoviews.plotting.bokeh
import numpy as np
import panel as pn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from wordcloud import WordCloud, STOPWORDS
from wordcloud import WordCloud
import wordcloud


#%%

def chord_chard(data):
    
    """
    
    Takes in processed dataframe for multilabel classification problem and computes label co-occurences.
    
    Draws chord chard using bokeh and local server.
    
    """
    hv.extension('bokeh')
    
    hv.output(size=200)
    
    #labels_only =  data.drop(labels = ['PIMS_ID', 'language', 'description', 'all_logs', 'text'], axis=1)
    
    labels_only =  data

    cooccurrence_matrix = np.dot(labels_only.transpose(),labels_only)
    
    cooccurrence_matrix_diagonal = np.diagonal(cooccurrence_matrix)
    with np.errstate(divide='ignore', invalid='ignore'):
        cooccurrence_matrix_percentage = np.nan_to_num(np.true_divide(cooccurrence_matrix, cooccurrence_matrix_diagonal[:, None]))
    
    coocc = labels_only.T.dot(labels_only)
    diagonal = np.diagonal(coocc)
    co_per = np.nan_to_num(np.true_divide(coocc, diagonal[:, None]))
    df_co_per = pd.DataFrame(co_per)
    df_co_per = pd.DataFrame(data=co_per, columns=coocc.columns, index=coocc.index)
    
    #replace diagonal with 0:
    coocc.values[[np.arange(coocc.shape[0])]*2] = 0
    coocc = coocc.mask(np.triu(np.ones(coocc.shape, dtype=np.bool_)))
    coocc = coocc.fillna(0)

    data = hv.Dataset((list(coocc.columns), list(coocc.index), coocc),
                      ['source', 'target'], 'value').dframe()
    data['value'] = data['value'].astype(int)
    chord = hv.Chord(data)

    plot = chord.opts(
        node_color='index', edge_color='source', label_index='index', 
        cmap='Category20', edge_cmap='Category20', width=400, height=400)
    

    bokeh_server = pn.Row(plot).show(port=1234)
    
def histogram(data, bins):
    '''
    Takes in processed dataframe and shows the number of occurrences of different values in the data
    ''' 
    data.hist(bins)


def Cramer_V(confusion_matrix):
    '''
    Shows correlation between categorical columns (like heat mao for numeric columns)
    '''
    import scipy.stats as ss
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    
def bubble_plot(df, cat1, cat2):
    '''
    Shows correlation between two specific categorical columns
    '''
    from bubble_plot.bubble_plot import bubble_plot
    bubble_plot(df, x=cat1, y=cat2)
    
def Outlier_analysis(data):
    '''
    finding outlier based on multiple columns using various algorithms such as Isolation forest
    '''
    from sklearn.ensemble import IsolationForest
    iso_forest = IsolationForest(n_estimators=300, contamination=0.10)
    iso_forest = iso_forest .fit(data)
    isof_outliers = iforest.predict(data)
    isoF_outliers_values = data[iforest.predict(data) == -1]
    
    return isoF_outliers_values
    
def Radar_chart():
    '''
    help in comparison (optional)
    '''
    
def nn_vis():
    '''
    Neural network visualisation can help understand what combination of columns could be important features or also to understand hidden or latent features.
    '''
    
def sankey_charts():
    '''
    Sankey charts can be very useful in making path analysis (between labels and categories)
    '''



def draw_cloud(dataframe, column):
    
    # Join the different processed titles together.
    long_string = ','.join(list(dataframe[column]))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=6, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    return wordcloud.to_image()