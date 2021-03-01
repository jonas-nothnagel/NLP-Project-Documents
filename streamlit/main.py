#basics
import pandas as pd
import numpy as np
#import joblib
import pickle5 as pickle
#import pickle
import time
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

#fuzzy search
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

#eli5
from eli5 import show_prediction

#Whoosh (ElasticSearch)
from whoosh.index import open_dir
from whoosh.qparser import MultifieldParser,OrGroup, query
from whoosh import scoring
from whoosh import highlight
#load indexed document storage and specify whoosh:
ix = open_dir("./whoosh/whoosh")
weighting_type = scoring.BM25F()
fields = ['all_text_clean']
og = OrGroup.factory(0.9) #bonus scaler
parser = MultifieldParser(fields, ix.schema, group = og)

#streamlit
import streamlit as st
import SessionState
from load_css import local_css
local_css("./streamlit/style.css")

DEFAULT = '< PICK A VALUE >'
def selectbox_with_default(text, values, default=DEFAULT, sidebar=False):
    func = st.sidebar.selectbox if sidebar else st.selectbox
    return func(text, np.insert(np.array(values, object), 0, default))

#helper functions
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
import src.clean_dataset as clean
#import src.vectorize_embed as em

#experimental Transformer based approaches: SLOW
def zero_shot_classification():
    
    nli_model = AutoModelForSequenceClassification.from_pretrained('joeddav/xlm-roberta-large-xnli')
    tokenizer = AutoTokenizer.from_pretrained('joeddav/xlm-roberta-large-xnli')
    
    return nli_model, tokenizer

@st.cache
def neuralqa():
    
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", 
                                              use_fast=False)
    model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", 
                                                          return_dict=False)

    return tokenizer, model

@st.cache(allow_output_mutation=True)
def sentence_transformer(sentences):
    
    for i in range(len(sentences)):
        sentences[i] = sentences[i].replace('_', ' ')
    for i in range(len(sentences)):
        sentences[i] = sentences[i].lstrip()
    
    embedder = SentenceTransformer('paraphrase-distilroberta-base-v1')
    
    corpus_embeddings = embedder.encode(sentences, convert_to_tensor=True)
    
    return embedder, corpus_embeddings

sys.path.pop(0)

#%%
#1. load in complete transformed and processed dataset for pre-selection and exploration purpose
df = pd.read_csv('./data/processed/taxonomy_final.csv')
df_columns = df.drop(columns=['PIMS_ID', 'all_text_clean', 'all_text_clean_spacy',  'hyperlink',
 'title',
 'leading_country',
 'grant_amount',
 'country_code',
 'lon',
 'lat'])
    
to_match = df_columns.columns.tolist()

#2. load parent dict
with open("./data/processed/parent_dict.pkl", 'rb') as handle:
    parents = pickle.load(handle)

#3. load sub category dict
with open("./data/processed/category_dict.pkl", 'rb') as handle:
    sub = pickle.load(handle)    
    
#4. Load Training Scores:
with open("./data/processed/tfidf_only_f1.pkl", 'rb') as handle:
    scores_dict = pickle.load(handle)     

#5. Load all categories as list:
with open("./data/processed/all_categories_list.pkl", 'rb') as handle:
    all_categories = pickle.load(handle)

#6. Load df with targets:
df_targets = pd.read_csv('./data/processed/taxonomy_final_targets.csv')
df_columns = df_targets.drop(columns=['PIMS_ID', 'all_text', 'all_text_clean', 'all_text_clean_spacy',  'hyperlink',
 'title',
 'leading_country',
 'grant_amount',
 'country_code',
 'lon',
 'lat'])
    
to_match_targets = df_columns.columns.tolist()
    
#sort list
all_categories = sorted(all_categories)    
#%%
session = SessionState.get(run_id=0)


#%%
#title start page
st.title('Machine Learning for Nature Climate Energy Portfolio')

sdg = Image.open('./streamlit/logo.png')
st.sidebar.image(sdg, width=200)
st.sidebar.title('Navigation')


#%%
# app decision:
app_selection = st.sidebar.selectbox('What application would you like to test?', 
                             ('', 'Elastic Search', 'Fuzzy Structured Search', 'ML Classification', 'Zero-Shot Classification',
                              'Neural Question Answering'),
                             format_func=lambda x: 'Default Data Exploration' if x == '' else x)

#%%
# define containers:
container_1 = st.empty()  
container_2 = st.empty()  
container_3 = st.empty()  
container_4 = st.empty()
container_5 = st.empty()
container_6 = st.empty()

#%%
#explore data:
if app_selection == '':
    container_2 = st.empty()  
    container_3 = st.empty()  
    container_4 = st.empty()    
    container_5 = st.empty()
    container_6 = st.empty()
    
    with container_1.beta_container():
        st.write('## Explore the portfolio with new taxonomy:')
        options = st.multiselect('Select categories from the taxonomy:', 
                                              all_categories, format_func=lambda x: 'Select a category' if x == '' else x)
        
        
        filters = []
        if options:
            for k in options:
                condition =(k,1)
                filters.append(condition)
            f = '{0[0]} == {0[1]}'.format
            result_df = df.query(' & '.join(f(t) for t in filters))
            st.write('Relevant Project IDs:', result_df[['PIMS_ID', 'title', 'leading_country', 'grant_amount']], "Number of projects:", len(result_df))
            result_df = result_df.loc[result_df['lat'] != 51.0834196]
            st.map(result_df)
    
#%%
#Sidebar category choosing

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
        
        agree = st.sidebar.checkbox(label='Would you like to display and predict sub-categories of your choice?', value = False)            
        if agree:
            sub_option = st.sidebar.selectbox('Select a category:', parents[option], format_func=lambda x: 'Select a category' if x == '' else x)
            if sub_option:
                st.sidebar.markdown("**Sub Categories:**")
                for i in sub[sub_option]:
                    st.sidebar.write(i)
            categories = sub[sub_option]                   
        else:
            categories = parents[option]
            
        #choose one category from all:
        agree = st.sidebar.checkbox(label='Would you like to predict specific categories?', value = False)            
        if agree:
            all_options = st.sidebar.multiselect('Select a category:', all_categories, format_func=lambda x: 'Select a category' if x == '' else x)
            if all_options:
                st.sidebar.markdown("**You've chosen:**")
                for i in all_options:
                    st.sidebar.write(i)                
                categories = all_options

        # predict each category:             
        agree = st.sidebar.checkbox(label='Would you like to predict the whole taxonomy?', value = False, key= "key1")            
        if agree:
            categories = all_categories                              
    else:
        st.warning('No category is selected')
        
#%%
#Container 1:  
if app_selection == 'ML Classification':
    
    #delete containers:
    container_1 = st.empty()  
    container_3 = st.empty()  
    container_4 = st.empty()    
    container_5 = st.empty()
    container_6 = st.empty()
    
    with container_2.beta_container():
        
        st.write('## Frontend Application that takes text as input and outputs classification decision.')
        
        model_selection = st.selectbox('What model architecture would you like to use?', ('TFIDF', 'TFIDF + LSA Dimension Reduction', 
                                                                                          'Transformer + Neural Networks'))
        st.write(model_selection)
        
        if model_selection == 'Transformer + Neural Networks':
            st.write("Under Construction...Please choose other option for now.")       
        else:
            pass
        
        text_input = st.text_input('Please Input your Text:')
        
        #define lists
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
                                    

            time.sleep(3)
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



   
# add elastic search after fuzzy structured search.
                                    
if app_selection == 'Fuzzy Structured Search':
    
    container_1 = st.empty()  
    container_2 = st.empty()  
    container_4 = st.empty()
    container_5 = st.empty()   
    container_6 = st.empty()
    
    with container_3.beta_container():
        
        model_selection = st.selectbox('Choose Semantic Search model:', ('Fuzzy Elastic Search', 'Neural Sentence Transformers'))
                
            
        if model_selection == "Fuzzy Elastic Search":
            input_query = st.text_input('Please Input your Query: (works best for keywords - coral reef, agroforestry, etc)')
            if input_query != '':
                matches = process.extract(input_query, to_match_targets, limit = 15)
                match_string = []
                for match in matches:
                    for m in match:
                        if type(m) == str:
                            match_string.append(m)
                        
        if model_selection == "Neural Sentence Transformers":            
            embedder, corpus_embeddings = sentence_transformer(to_match_targets)
            
            input_query = st.text_input('Please Input your Query: (works best for keywords - coral reef, agroforestry, etc)')
            if input_query != '':            
                query_embedding = embedder.encode(input_query, convert_to_tensor=True)
                hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=15)
                hits = hits[0]
                match_string = []
                for hit in hits:
                    match_string.append(to_match_targets[hit['corpus_id']])
                
        if input_query != '': 
            options = st.multiselect('These are the most likely categories. Select:', 
                match_string, format_func=lambda x: 'Select a category' if x == '' else x)
           
            filters = []
            if options:
                for k in options:
                    condition =(k,1)
                    filters.append(condition)
                f = '{0[0]} == {0[1]}'.format
                
                col1, col2 = st.beta_columns(2)
                and_search = col1.checkbox(label='AND', value = False)            
                if and_search:                
                    result_df = df_targets.query(' & '.join(f(t) for t in filters))
                    st.write('Relevant Project IDs:', result_df[['PIMS_ID', 'title', 'leading_country', 'grant_amount']], "Number of projects:", len(result_df))
                    result_df = result_df.loc[result_df['lat'] != 51.0834196]
                    #st.map(result_df)
                or_search = col2.checkbox(label='OR', value = True)
                if or_search:
                    result_df = df_targets.query(' or '.join(f(t) for t in filters))
                    st.write('Relevant Project IDs:', result_df[['PIMS_ID', 'title', 'leading_country', 'grant_amount']], "Number of projects:", len(result_df))
                    result_df = result_df.loc[result_df['lat'] != 51.0834196]
        
                pims_structured = result_df.PIMS_ID.tolist()
                
                if len(pims_structured) > 0:
                    
                    #Compare with ElasticSearch:
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
                    recall = len(intersections)/len(pims)
                    st.write('Number of intersections with ElasticSearch results:', len(intersections))
                    #st.write('Percentage of relevant projects found by ElasticSearch:', round(intersection_ratio,2)*100, "%")
                    st.write('Recall of ElasticSearch:', round(recall,2)*100, "%")
                    st.map(result_df)
                
if app_selection == 'Elastic Search':
    
    container_1 = st.empty()  
    container_2 = st.empty()  
    container_3 = st.empty()  
    container_5 = st.empty()
    container_6 = st.empty()
    
    with container_4.beta_container():
        

        input_query = st.text_input('Please Input your Text:')
        returns = st.slider('How many documents you want to return?', 1, 500)
        
        q = parser.parse(input_query)


        pims = []
        title = []
        grant_amount = []
        leading_country = []
        with ix.searcher(weighting = weighting_type) as s:
          results = s.search(q, limit = returns)
          results.fragmenter = highlight.SentenceFragmenter()
          results.formatter = highlight.UppercaseFormatter()
          
          for res in results:
             pims.append(res['PIMS_ID'])
             pims = [ int(x) for x in pims ]
             title.append(res['title'])
             grant_amount.append(res['grant_amount'])             
             leading_country.append(res['leading_country'])
        result = dict(zip(pims, title))
        result = pd.DataFrame(result.items(), columns=['PIMS_ID', 'title'])
        result= result.assign(leading_country=leading_country)
        result= result.assign(grant_amount=grant_amount)
        
        result['PIMS_ID'] = result['PIMS_ID'].astype(int)
        mapping = df[['PIMS_ID', 'lon', 'lat']]
        result = result.merge(mapping, how='left', on=['PIMS_ID'])
        
        st.write('Relevant Project IDs:', result[['PIMS_ID', 'title', 'leading_country', 'grant_amount']], "Number of projects:", len(result))
                
        result = result.loc[result['lat'] != 51.0834196]
        st.map(result)
        
if app_selection == "Zero-Shot Classification":
    
    container_1 = st.empty()  
    container_2 = st.empty()  
    container_3 = st.empty()  
    container_4 = st.empty()
    container_6 = st.empty()
    
    with container_5.beta_container():
        st.write("Try unsupervised text classification leveraging state-of-the-Art language models. WARNING: SLOW")
        sequence = st.text_input('Please Specify your sentence/text:')
        
        if sequence != "":
            st.write("Sequences are posed as the premises and topic labels are turned into premises, i.e. biodiversity -> This text is about biodiversity.")
            candidate_labels = st.text_input('Now please specify categories (comma seperated)')
    
            if candidate_labels != "":
                candidate_labels = candidate_labels.split(",")
        

                with st.spinner('Load Weights of Transformer model...'):
                    nli_model, tokenizer = zero_shot_classification()
                
                    premise = sequence
                    results = []
                    for label in candidate_labels:
                        hypothesis = f'This example is {label}.'
                        
                        # run through model pre-trained on MNLI
                        x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                                             truncation_strategy='only_first')
                        logits = nli_model(x)[0]
            
                        entail_contradiction_logits = logits[:,[0,2]]
                        probs = entail_contradiction_logits.softmax(dim=1)
                        prob_label_is_true = probs[:,1]
                        results.append(round(prob_label_is_true.item(),3))
        
                    result = dict(zip(candidate_labels, results))
                    result = pd.DataFrame({'labels': candidate_labels, 'scores': results})
                    
                    state_total_graph = px.bar(
                    result, 
                    x='labels',
                    y='scores',
                    color ="labels")
                    st.plotly_chart(state_total_graph)                    

        
                    
# 
if app_selection == "Neural Question Answering":
    
    container_1 = st.empty()  
    container_2 = st.empty()  
    container_3 = st.empty()  
    container_4 = st.empty()
    container_5 = st.empty()       
    
    with container_6.beta_container():
        st.write("Try Neural Question Answering. PROTOTYPE.")
        
        examples=["", 
                  "what's the problem with the artibonite river basin?", 
                  "what are threads for the machinga and mangochi districts of malawi?",
                  "how can we deal with rogue swells?"]
        
        example = st.selectbox('Examples:', [k for k in examples], format_func=lambda x: 'Select an Example' if x == '' else x)
              
        question = st.text_input('Try own question (be as specific as possible):')
        
        #load and split dataframe:
        splitted = clean.split_at_length(df, 'all_text_clean', 500)
        
        if example != "":
            
            question = example
            
        if question != "":
            
            question = question
            
            with st.spinner('Running ElasticSearch to find relevant documents...'):
                ix = open_dir("../whoosh/split")
                weighting_type = scoring.BM25F()
                fields = ['text']
                og = OrGroup.factory(0.9) #bonus scaler
                parser = MultifieldParser(fields, ix.schema, group = og)
                
                q = parser.parse(question)
                
                pims = []
                all_text = []
                #title = []
                #grant_amount = []
                #leading_country = []
                
                with ix.searcher(weighting = weighting_type) as s:
                  results = s.search(q, limit = 1)
                  results.fragmenter = highlight.SentenceFragmenter()
                  results.formatter = highlight.UppercaseFormatter()
                  
                  for res in results:
                     pims.append(res['PIMS_ID'])
                     all_text.append(res['text'])
                     pims = [ int(x) for x in pims ]
                     #title.append(res['title'])
                     #grant_amount.append(res['grant_amount'])             
                     #leading_country.append(res['leading_country'])
                     
                result = dict(zip(pims, all_text))
                result = pd.DataFrame(result.items(), columns=['PIMS_ID', 'text'])
                #result= result.assign(text=all_text)
                #result= result.assign(leading_country=leading_country)
                #result= result.assign(grant_amount=grant_amount)
                
                result['PIMS_ID'] = result['PIMS_ID'].astype(int)
                
                #texts = []
                #for i in result.text:
                    #texts.append(str(i))
                
                text = str(result.text.iloc[0])
                
                st.write(result)
            
                if text != "":
                    with st.spinner('Load Weights of Transformer model...'):
                        tokenizer, model = neuralqa()
                            
                        inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
                        input_ids = inputs["input_ids"].tolist()[0]
        
                        text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
                        answer_start_scores, answer_end_scores = model(**inputs)
        
                        answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
                        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
        
                        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        
                        st.write(f"Question: {question}")
                        
                        st.write(f"Answer: {answer}\n")
                        
                        st.markdown("**Context:**")
                        
                        green = "<span class='highlight green'>"+answer+"</span>"
                        
                        text = text.replace(answer, green)
                        text = "<div>"+text+"</div>"
                        st.markdown(text, unsafe_allow_html=True)
                        
#%%
st.write('           ')
st.write('           ')
st.write('           ')
st.write('           ')
st.write('           ')
st.write('           ')
if st.button("Run again!"):
  session.run_id += 1

