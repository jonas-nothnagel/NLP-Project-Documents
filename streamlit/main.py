#basics
import pandas as pd
import numpy as np
from inspect import getsourcefile
import os
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
import joblib
import pickle
import time
from PIL import Image

#fuzzy search
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


#eli5
import eli5
from eli5 import explain_weights, explain_prediction, show_prediction
from eli5.formatters import format_as_html, format_as_text, format_html_styles

#Whoosh (ElasticSearch)
from whoosh.index import create_in
from whoosh.index import open_dir
from whoosh.fields import Schema, TEXT
from whoosh.qparser import MultifieldParser,OrGroup, query
from whoosh import scoring
from whoosh import highlight
        #load indexed document storage and specify whoosh:
ix = open_dir("../whoosh")
weighting_type = scoring.BM25F()
fields = ['all_text']
og = OrGroup.factory(0.9) #bonus scaler
parser = MultifieldParser(fields, ix.schema, group = og)

#streamlit
import streamlit as st
from annotated_text import annotated_text
import SessionState
from load_css import local_css
local_css("style.css")

DEFAULT = '< PICK A VALUE >'
def selectbox_with_default(text, values, default=DEFAULT, sidebar=False):
    func = st.sidebar.selectbox if sidebar else st.selectbox
    return func(text, np.insert(np.array(values, object), 0, default))

#helper functions
import src.clean_dataset as clean
#import src.vectorize_embed as em

sys.path.pop(0)






#%%
#1. load in complete transformed and processed dataset for pre-selection and exploration purpose
df = pd.read_csv('../data/processed/taxonomy_final.csv')
df_columns = df.drop(columns=['PIMS_ID', 'all_text_clean', 'all_text_clean_spacy',  'hyperlink',
 'title',
 'leading_country',
 'grant_amount',
 'country_code'])
to_match = df_columns.columns.tolist()


#2. load parent dict
with open("../data/processed/parent_dict.pkl", 'rb') as handle:
    parents = pickle.load(handle)

#3. load sub category dict
with open("../data/processed/category_dict.pkl", 'rb') as handle:
    sub = pickle.load(handle)    
    
#4. Load Training Scores:
with open("../data/processed/tfidf_only_f1.pkl", 'rb') as handle:
    scores_dict = pickle.load(handle)     

#5. Load all categories as list:
with open("../data/processed/all_categories_list.pkl", 'rb') as handle:
    all_categories = pickle.load(handle) 
    
#%%
session = SessionState.get(run_id=0)
    
#%%
#title page
st.title('Machine Learning for Nature Climate Energy Portfolio')

#title sidebar
image = Image.open('logo.png')
st.sidebar.image(image, width=150)
st.sidebar.title('Navigation')
#%%
# app decision:
app_selection = st.sidebar.selectbox('What application would you like to test?', 
                             ('', 'Elastic Search', 'Fuzzy Structured Search', 'ML Classification'),
                             format_func=lambda x: 'Select an Option' if x == '' else x)

# define containers:
container_1 = st.empty()  
container_2 = st.empty()  
container_3 = st.empty()  

#%%
# Sidebar category choosing

if app_selection == 'ML Classification':
    
    items = [k  for  k in  parents.keys()]
    items.insert(0,'')
    option = st.sidebar.selectbox('Select a category:', items, format_func=lambda x: 'Select a category' if x == '' else x)
    if option:
        st.sidebar.write('You selected:', option)
        st.sidebar.markdown("**Categories**")

        for i in parents[option]:
            st.sidebar.write(i)
        st.sidebar.markdown("**Further Choices**")
        
        agree = st.sidebar.checkbox(label='Would you like to display and predict sub-categories?', value = False)            
        if agree:
            sub_option = st.sidebar.selectbox('Select a category:', parents[option], format_func=lambda x: 'Select a category' if x == '' else x)
            if sub_option:
                st.sidebar.markdown("**Sub Categories:**")
                for i in sub[sub_option]:
                    st.sidebar.write(i)
            categories = sub[sub_option]                   
        else:
            categories = parents[option]
            
        agree = st.sidebar.checkbox(label='Would you like to predict the whole taxonomy?', value = False)            
        if agree:
            
            categories = all_categories                  
        else:
            categories = parents[option]
            
    else:
        st.warning('No category is selected')
#%%
#Container 1:  
if app_selection == 'ML Classification':
    
    # delete containers:
    container_2 = st.empty()  
    container_3 = st.empty()  
    
    with container_1.beta_container():
        
        st.write('## Frontend Application that takes text as input and outputs classification decision.')
        
        model_selection = st.selectbox('What model architecture would you like to use?', ('TFIDF', 'TFIDF + LSA Dimension Reduction', 
                                                                                          'Transformer + Neural Networks'))
        st.write(model_selection)
        
        if model_selection == 'Transformer + Neural Networks':
            st.write("Under Construction...Please choose other option for now.")       
        else:
            pass
        
        # Prediction Function
        text_input = st.text_input('Please Input your Text:')
        
        name = []
        hat = []
        number = []
        
        top_5 = []
        last_5 = []
        top_5_10 = []
        last_5_10 = []
        
        if text_input != '':
            placeholder = st.empty()
            
            with placeholder.beta_container():
                with st.spinner('Load Models and Predict...'):
                    for category in categories:
                        
                        # take only models with over 20 training datapoints
                        if df[category].sum(axis=0) > 20:
                            
                            #prune the long names to ensure proper loading
                            if len(category) > 50:
                                category = category[0:20] 
                                st.write(category)
                            else:
                                pass 
                            
                            # Pre-process text:
                            input_list = [text_input]
                            input_df = pd.DataFrame(input_list, columns =['input_text'])
                            
                            # clean text
                            input_df['input_text'] = input_df['input_text'].apply(clean.spacy_clean)
                            clean_df = pd.Series(input_df['input_text'])   
                          
                            if model_selection == 'TFIDF':

                                tfidf_vectorizer = joblib.load('../models/tf_idf/tf_idf_only/'+category+'_'+'vectorizer.sav')        
                                fnames = tfidf_vectorizer.get_feature_names()
                                
                                vector_df = tfidf_vectorizer.transform(clean_df)

                                clf = joblib.load('../models/tf_idf/tf_idf_only/'+category+'_'+'model.sav')
                                y_hat = clf.predict(vector_df)
                                y_prob = clf.predict_proba(vector_df)

                                
                                if y_hat == 1:        
                                    element = st.write(category)
                                    number.append(df[category].sum(axis=0))
                                    name.append(category)
                                    element = st.write("Yes with Confidence:", y_prob[0][1].round(2)*100, "%")                  
                                    hat.append(y_prob[0][1].round(2)*100)
                                    
                                    results= dict(zip(name, hat))
                                    
                                    #return top features:
                                    w = show_prediction(clf, tfidf_vectorizer.transform(clean_df), 
                                                    highlight_spaces = True, 
                                                    top=5000, 
                                                    feature_names=fnames, 
                                                    show_feature_values  = True)                                    
                                    result = pd.read_html(w.data)[0]
                                    top_5_list = result.Feature.iloc[0:5].tolist()
                                    top_5.append(top_5_list)
                                    
                                    top_5_10_list = result.Feature.iloc[5:10].tolist()
                                    top_5_10.append(top_5_10_list)
                                    
                                    last_5_list = result.Feature.iloc[-5:].tolist()
                                    last_5.append(last_5_list)
                                    
                                    last_5_10_list = result.Feature.iloc[-10:].tolist()
                                    last_5_10_list = list(set(last_5_10_list) - set(last_5_list))
                                    last_5_10.append(last_5_10_list)
                                    
                                    
                                    
                                    
                                if y_hat == 0:
                                    element= st.write(category)
                                    element = st.write("No with Confidence:", y_prob[0][0].round(2)*100, "%")
                                
                            if model_selection == 'TFIDF + LSA Dimension Reduction':

                                
                                tfidf_vectorizer = joblib.load('../models/tf_idf/tf_idf_lsa/'+category+'_'+'vectorizer.sav')        
                                lsa = joblib.load('../models/tf_idf/tf_idf_lsa/'+category+'_'+'lsa.sav')
                                
                                vector_df = tfidf_vectorizer.transform(clean_df)
                                vector_df = lsa.transform(vector_df)
                        
                                clf = joblib.load('../models/tf_idf/tf_idf_lsa/'+category+'_'+'model.sav')
                                y_hat = clf.predict(vector_df)
                                y_prob = clf.predict_proba(vector_df)
                                
                                
                                if y_hat == 1:        
                                    element = st.write(category)
                                    name.append(category)
                                    element = st.write("Yes with Confidence:", y_prob[0][1].round(2)*100, "%")                  
                                    hat.append(y_prob[0][1].round(2)*100)
                                    
                                    results= dict(zip(name, hat))
                                    
                                if y_hat == 0:
                                    element= st.write(category)
                                    element = st.write("No with Confidence:", y_prob[0][0].round(2)*100, "%")
                                    

            #time.sleep(2)
            placeholder.empty()
            if name != []:    
                t = "<div> <span class='highlight green'>Suggested Categories:</div>"
                st.markdown(t, unsafe_allow_html=True)
                st.write('           ')
                
                
                a = 0
                for key, value in results.items():
                    new_df = clean_df
                    st.write(key, 'with', value, '% confidence.')
                    st.write('Model was trained on', number[0], 'examples with accuracy (F1 Score) of:', scores_dict[key].round(2)*100, "%")
                    
                    if model_selection == "TFIDF":
                        
                        st.write('Detailed Explanation of Prediction:')
                        for item in top_5[a]:
                            green = "<span class='highlight green'>"+item+"</span>"
                            item = item
                            new_df = new_df.str.replace(item,green)
                            
                        for item in last_5[a]:    
                            red = "<span class='highlight red'>"+item+"</span>"
                            item = " "+item+" "
                            new_df = new_df.str.replace(item,red)
                            
                        for item in top_5_10[a]:
                            lightgreen = "<span class='highlight lightgreen'>"+item+"</span>"
                            item = " "+item+" "
                            new_df = new_df.str.replace(item,lightgreen)
                            
                        for item in last_5_10[a]:
                            lightred = "<span class='highlight IndianRed'>"+item+"</span>"
                            item = " "+item+" "
                            new_df = new_df.str.replace(item,lightred)
        
                        text = new_df[0]
                        text = "<div>"+text+"</div>"
                        st.markdown(text, unsafe_allow_html=True)
                        st.write('           ')
                        st.write('           ')
                        st.write('           ')
                        
                        a = a+1
            else:
                t = "<div> <span class='highlight red'>Not enough confidence in any category.</div>"
                st.markdown(t, unsafe_allow_html=True)        

if app_selection == 'Fuzzy Structured Search':
    
    container_1 = st.empty()  
    container_3 = st.empty()  
    with container_2.beta_container():
        
        input_query = st.text_input('Please Input your Text:')
        
        if input_query != '':
            #words = input_query.split()
            #query = words[-1]
            matches = process.extract(input_query, to_match, limit = 5)
            match_string = []
            for match in matches:
                for m in match:
                    if type(m) == str:
                        match_string.append(m)
            
            option = st.selectbox("These are the most likely categories. Select:" ,match_string)
            st.write('you selected', option)
                
                
            if option in df.columns.tolist():                
                result = df.loc[df[option] == 1]
                result = result.reset_index(drop=True)
                result = result[['PIMS_ID', 'title', 'hyperlink', 'lon', 'lat']]   
                summary = dict(zip(result.PIMS_ID.tolist(),result.hyperlink.tolist()))
                
                # make clickable and link to document:
                st.write('Relevant Project IDs:', result[['PIMS_ID', 'title']], "Number of projects:", len(result))
                
                pims_structured = result.PIMS_ID.tolist()
                # compare with ElasticSearch:
                q = parser.parse(input_query)
                
                pims = []
                with ix.searcher(weighting = weighting_type) as s:
                    el_results = s.search(q, limit = 1000)
                    el_results.fragmenter = highlight.SentenceFragmenter()
                    el_results.formatter = highlight.UppercaseFormatter()
                    for res in el_results:
                        pims.append(res['PIMS_ID'])
                        pims = [ int(x) for x in pims ]
                        length = len(pims)

                intersections = list(set(pims_structured) & set(pims))
                intersection_ratio = len(intersections)/len(pims_structured)
                st.write('Number of intersections with ElasticSearch results:', len(intersections))
                st.write('Percentage of relevant projects found by ElasticSearch:', round(intersection_ratio,2)*100, "%")
                
                result = result.loc[result['lat'] != 51.0834196]
                st.map(result)


                                       
            else:
                st.write('No good matches')

            # options = st.multiselect("Combine Categories:" ,match_string)
            # st.write('you selected', options)

                
if app_selection == 'Elastic Search':
    
    container_1 = st.empty()  
    container_2 = st.empty()  
    with container_2.beta_container():
        

        input_query = st.text_input('Please Input your Text:')
        returns = st.slider('How many documents you want to return?', 1, 100)
        
        q = parser.parse(input_query)


        pims = []
        title = []
        with ix.searcher(weighting = weighting_type) as s:
          results = s.search(q, limit = returns)
          results.fragmenter = highlight.SentenceFragmenter()
          results.formatter = highlight.UppercaseFormatter()
          for res in results:
             pims.append(res['PIMS_ID'])
             pims = [ int(x) for x in pims ]
             title.append(res['title'])
        result = dict(zip(pims, title))
        result = pd.DataFrame(result.items(), columns=['PIMS_ID', 'title'])
        result['PIMS_ID'] = result['PIMS_ID'].astype(int)
        mapping = df[['PIMS_ID', 'lon', 'lat']]
        result = result.merge(mapping, how='left', on=['PIMS_ID'])
        
        st.write('Relevant Project IDs:', result[['PIMS_ID', 'title']], "Number of projects:", len(result))
                
        result = result.loc[result['lat'] != 51.0834196]
        st.map(result)
#%%
st.write('           ')
st.write('           ')
st.write('           ')
st.write('           ')
st.write('           ')
st.write('           ')
if st.button("Run again!"):
  session.run_id += 1

