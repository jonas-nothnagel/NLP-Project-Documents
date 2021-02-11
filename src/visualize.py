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
import re
import umap.umap_ as umap
from wordcloud import WordCloud, STOPWORDS
from wordcloud import WordCloud
import wordcloud

from collections import Counter, defaultdict

#%%
def disp_category(category, df, min_items):
    '''
    Display Categorical charts
  
    '''
    dff = df.groupby(category)
    class_ = dff.count().sort_values(by='PIMS_ID')['PIMS_ID'].reset_index()
    class_.columns=[category,'count']
    
    class_= class_[class_['count']>=min_items][category]
    df = df[df[category].isin(class_)]

    labels = df[category]
    
    counts = defaultdict(int)
    for l in labels:
         counts[l] += 1

    counts_df = pd.DataFrame.from_dict(counts, orient='index')
    counts_df.columns = ['count']
    counts_df.sort_values('count', ascending=False, inplace=True)

    fig, ax = plt.subplots()
    ax = sns.barplot(x=counts_df.index, y=counts_df['count'], ax=ax)
    fig.set_size_inches(10,5)
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=-90);
    ax.set_title('Label: '+category.upper())
    plt.show()
    return class_


#%%
def scatter(category, df, min_items):
    '''
    Display scatterplot 
  
    '''
    dff = df.groupby(category)
    class_ = dff.count().sort_values(by='PIMS_ID')['PIMS_ID'].reset_index()
    class_.columns=[category,'count']
    
    class_= class_[class_['count']>=min_items][category]
    df = df[df[category].isin(class_)]

    labels = df[category]
    
    counts = defaultdict(int)
    for l in labels:
         counts[l] += 1

    counts_df = pd.DataFrame.from_dict(counts, orient='index')
    counts_df.columns = ['count']
    counts_df.sort_values('count', ascending=False, inplace=True)

    fig, ax = plt.subplots()
    sns.despine(fig, left=True, bottom=True)
   
    ax = sns.scatterplot(x=counts_df.index, y=counts_df['count'],
               
                palette="ch:r=-.2,d=.3_r",
                linewidth=0,
                data=df, ax=ax)
 

    ax.set_title('Label: '+category.upper())
    plt.show()
    return class_


#%%

import plotly.graph_objs as go
from plotly.offline import iplot

def plotly_bar(category, df, min_items):
    '''
    Display Categorical charts
  
    '''
    dff = df.groupby(category)
    class_ = dff.count().sort_values(by='PIMS_ID')['PIMS_ID'].reset_index()
    class_.columns=[category,'count']
    
    class_= class_[class_['count']>=min_items][category]
    df = df[df[category].isin(class_)]

    labels = df[category]
    
    counts = defaultdict(int)
    for l in labels:
         counts[l] += 1

    counts_df = pd.DataFrame.from_dict(counts, orient='index')
    counts_df.columns = ['count']
    counts_df.sort_values('count', ascending=False, inplace=True)
    data = [go.Bar(
       x = counts_df.index,
       y = counts_df['count']
    )]
    fig = go.Figure(data=data)
    iplot(fig)
    
    
    return fig

def plotly_pie(category, df, min_items):
    '''
    Display Categorical charts
  
    '''
    dff = df.groupby(category)
    class_ = dff.count().sort_values(by='PIMS_ID')['PIMS_ID'].reset_index()
    class_.columns=[category,'count']
    
    class_= class_[class_['count']>=min_items][category]
    df = df[df[category].isin(class_)]

    labels = df[category]
    
    counts = defaultdict(int)
    for l in labels:
         counts[l] += 1

    counts_df = pd.DataFrame.from_dict(counts, orient='index')
    counts_df.columns = ['count']
    counts_df.sort_values('count', ascending=False, inplace=True)
    data = [go.Pie(
 
       labels = counts_df.index,
       values = counts_df['count'],
       
    )]
    fig = go.Figure(data=data)
    iplot(fig)
    
    return fig

def plotly_scatter(category, df, min_items):
    '''
    Display Categorical charts
  
    '''
    dff = df.groupby(category)
    class_ = dff.count().sort_values(by='PIMS_ID')['PIMS_ID'].reset_index()
    class_.columns=[category,'count']
    
    class_= class_[class_['count']>=min_items][category]
    df = df[df[category].isin(class_)]

    labels = df[category]
    
    counts = defaultdict(int)
    for l in labels:
         counts[l] += 1

    counts_df = pd.DataFrame.from_dict(counts, orient='index')
    counts_df.columns = ['count']
    counts_df.sort_values('count', ascending=False, inplace=True)
    
    trace0 = go.Scatter(
       x = counts_df.index,
       y = counts_df['count'],
       mode = 'markers',
       name = 'markers'
    )
    trace1 = go.Scatter(
       x = counts_df.index,
       y = counts_df['count'],
       mode = 'lines+markers',
       name = 'line+markers'
    )
    trace2 = go.Scatter(
       x = counts_df.index,
       y = counts_df['count'],
       mode = 'lines',
       name = 'line'
    )
    data = [trace0, trace1, trace2]
    fig = go.Figure(data=data)
    iplot(fig)

    
    
    return fig

def plotly_radar(category, df, min_items):
    '''
    Display Categorical charts
  
    '''
    dff = df.groupby(category)
    class_ = dff.count().sort_values(by='PIMS_ID')['PIMS_ID'].reset_index()
    class_.columns=[category,'count']
    
    class_= class_[class_['count']>=min_items][category]
    df = df[df[category].isin(class_)]

    labels = df[category]
    
    counts = defaultdict(int)
    for l in labels:
         counts[l] += 1

    counts_df = pd.DataFrame.from_dict(counts, orient='index')
    counts_df.columns = ['count']
    counts_df.sort_values('count', ascending=False, inplace=True)
    
    radar = go.Scatterpolar(
       r = counts_df['count'],
       theta = counts_df.index,
       fill = 'toself'
    )
    data = [radar]
    fig = go.Figure(data=data)
    iplot(fig)
    
    
    return fig

def chord_chard(data):
    
    """
    
    Takes in processed dataframe for multilabel classification problem and computes label co-occurences.
    
    Draws chord chard using bokeh and local server.
    
    """
    hv.extension('bokeh')
    
    hv.output(size=200)
    
    labels_only =  data.drop(labels = ['PIMS_ID', 'project_description', 'logframe', 'all_text'], axis=1)
  

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
    

def genSankey(df,cat_cols=[],value_cols='',title='Sankey Diagram'):
    # maximum of 6 value cols -> 6 colors
    colorPalette = ['#4B8BBE','#306998','#FFE873','#FFD43B','#646464']
    labelList = []
    colorNumList = []
    for catCol in cat_cols:
        labelListTemp =  list(set(df[catCol].values))
        colorNumList.append(len(labelListTemp))
        labelList = labelList + labelListTemp
        
    # remove duplicates from labelList
    labelList = list(dict.fromkeys(labelList))
    
    # define colors based on number of levels
    colorList = []
    for idx, colorNum in enumerate(colorNumList):
        colorList = colorList + [colorPalette[idx]]*colorNum
        
    # transform df into a source-target pair
    for i in range(len(cat_cols)-1):
        if i==0:
            sourceTargetDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
            sourceTargetDf.columns = ['source','target','count']
        else:
            tempDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
            tempDf.columns = ['source','target','count']
            sourceTargetDf = pd.concat([sourceTargetDf,tempDf])
        sourceTargetDf = sourceTargetDf.groupby(['source','target']).agg({'count':'sum'}).reset_index()
        
    # add index for source-target pair
    sourceTargetDf['sourceID'] = sourceTargetDf['source'].apply(lambda x: labelList.index(x))
    sourceTargetDf['targetID'] = sourceTargetDf['target'].apply(lambda x: labelList.index(x))
    
    # creating the sankey diagram
    data = dict(
        type='sankey',
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(
            color = "black",
            width = 0.5
          ),
          label = labelList,
          color = colorList
        ),
        link = dict(
          source = sourceTargetDf['sourceID'],
          target = sourceTargetDf['targetID'],
          value = sourceTargetDf['count']
        )
      )
    
    layout =  dict(
        title = title,
        font = dict(
          size = 10
        )
    )
       
    fig = dict(data=[data], layout=layout)
    return fig

def draw_cloud(dataframe, column):
    
    # Join the different processed titles together.
    long_string = ','.join(list(dataframe[column]))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=6, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    return wordcloud.to_image()


def plot_proj(embedding, lbs):
    """
    Plot UMAP embeddings
    :param embedding: UMAP (or other) embeddings
    :param lbs: labels
    """
    n = len(embedding)
    counter = Counter(lbs)
    for i in range(len(np.unique(lbs))):
        plt.plot(embedding[:, 0][lbs == i], embedding[:, 1][lbs == i], '.', alpha=0.5,
                 label='cluster {}: {:.2f}%'.format(i, counter[i] / n * 100))
    plt.legend()
