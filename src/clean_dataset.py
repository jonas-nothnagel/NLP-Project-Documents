#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:56:48 2020

@author: jonas & oumaima

@title: clean_dataset

@descriptions: set of functions that enable different level of data cleaning.
"""

#%%
import pandas as pd
import numpy as np
import string  
import nltk 
#nltk.download('all')
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import re
import nltk
from nltk.tokenize import word_tokenize
from language_detector import detect_language
import pkg_resources
import spacy
import en_core_web_sm
import tqdm
from nltk.corpus import stopwords
import regex as re
import string
from itertools import chain
from textwrap import wrap
from langdetect.lang_detect_exception import LangDetectException
from langdetect import detect_langs

#from easynmt import EasyNMT
#model = EasyNMT('opus-mt')

from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
if sym_spell.word_count:
    pass
else:
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
sp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])

nltk.download('stopwords')
nltk.download('punkt')


# spacy (362 words)
spacy_st = nlp.Defaults.stop_words
# nltk(179 words)
nltk_st = stopwords.words('english') 
#%%

# text level preprocess 
# lowercase + base filter
# some basic normalization


def basic(s):
    """
    :param s: string to be processed
    :return: processed string: see comments in the source code for more info
    """
    # Text Lowercase
    s = s.lower() 
    # Remove punctuation
    translator = str.maketrans(' ', ' ', string.punctuation) 
    s = s.translate(translator)
    # Remove URLs
    s = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', s, flags=re.MULTILINE)
    s = re.sub(r"http\S+", " ", s)
    # Remove new line characters
    s = re.sub('\n', ' ', s) 
  
    # Remove distracting single quotes
    s = re.sub("\'", " ", s) 
    # Remove all remaining numbers and non alphanumeric characters
    s = re.sub(r'\d+', ' ', s) 
    s = re.sub(r'\W+', ' ', s)

    # define custom words to replace:
    #s = re.sub(r'strengthenedstakeholder', 'strengthened stakeholder', s)
    
    return s.strip()

def cat(s):
    
    # Text Lowercase
    s = s.lower() 
    # Remove whitespace at right:
    s = s.rstrip()
    # Remove punctuation
    s = re.sub(r'\W+', '_', s)
    # strip trailing _
    s = s.rstrip('_')
    return s


# language detection
def lang(s):
    """
    :param s: string to be processed
    :return: boolean (s is English)
    """
    # some reviews are actually english but biased toward french
    return detect_language(s) in {'English', 'French'}



# word level preprocess
# filtering out punctuations and numbers
def punct(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with punct and number filter out
    """
    return [word for word in w_list if word.isalpha()]


# selecting nouns
def noun(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with only nouns selected
    """
    return [word for (word, pos) in nltk.pos_tag(w_list) if pos[:2] == 'NN']


# typo correction
def typo(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with typo fixed by symspell. words with no match up will be dropped
    """
    w_list_fixed = []
    for word in w_list:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=3)
        if suggestions:
            w_list_fixed.append(suggestions[0].term)
        else:
            pass
            # do word segmentation, deprecated for inefficiency
            # w_seg = sym_spell.word_segmentation(phrase=word)
            # w_list_fixed.extend(w_seg.corrected_string.split())
    return w_list_fixed


# stemming if doing word-wise
p_stemmer = PorterStemmer()


def stem(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with stemming
    """
    return [p_stemmer.stem(word) for word in w_list]


# filtering out stop words
# create English stop words list
# We can add custom stopwords list later

en_stop = get_stop_words('en')
def stop(w_list):
    """
    filtering out stop words
    """
    return [word for word in w_list if word not in en_stop]


def preprocess_text(rw):
    """
    Get text level preprocessed data from raw review texts
    :param rw: review to be processed
    :return: text level pre-processed review
    """
    s = basic(rw)
    if not lang(s):
        return None
    return s

#This function is for bag-of-words models
def preprocess_word(s):
    """
    Get word level preprocessed data from preprocessed sentences
    including: remove punctuation, select noun, fix typo, stem, stop_words
    :param s: text string to be processed
    :return: word level pre-processed review
    """
    if not s:
        return None
    w_list = word_tokenize(s)
    w_list = punct(w_list)
    w_list = noun(w_list)
    #w_list = typo(w_list)
    w_list = stem(w_list)
    w_list = stop(w_list)

    return w_list

def clean(dataframe, column_name, basic_cleaning = True,
          stopwords = False, custom_stopwords = False, stemming = False, 
          lemmatizing = False, tokenizing = False):
    """
    Allows to input both pre-set english stopwords and costum stopword_list
    
    Probably employs spacy and nltk, has to allow for different levels.
    
    Outputs original data_frame with new_cleaned_column.
    """
    
    # let s be the string to be cleaned
    texts = column_name.to_list()
    clean_texts = []
    for s in texts:
        if not s:
            return None

        if basic_cleaning:
            s = preprocess_text(s)

        if tokenizing:
            w_list = word_tokenize(s)
            dataframe["words list"] = w_list

        if stemming:
            w_list = word_tokenize(s)
            w_list = stem(w_list)
            s = ' '.join(w_list)

        if stopwords:
            w_list = word_tokenize(s)
            w_list = stop(w_list)
            s = ' '.join(w_list)
            
        clean_texts.append(s)     
    
    dataframe['new_cleaned_column'] = clean_texts
      
    
def segment(dataframe, column_name):
    
    '''   
    optional: functions for segmenting:
        
        - costum sentence segmenting (spacy?) and re-merging with costum length (for BERT etc)
        - paragraph splitting (at \n etc)
    
    '''
    segments = spacy.load("en_core_web_sm")
    list_sentences = []
    texts = column_name.to_list()
    for s in texts:  
        sentences = []
        doc = segments(s) 
        for sent in doc.sents: 
            sentences.append(str(sent))
        list_sentences.append(sentences)
    dataframe['list_sentences'] = list_sentences 
    
def spacy_clean(alpha:str, use_nlp:bool = True) -> str:

    """

    Clean and tokenise a string using Spacy. Keeps only alphabetic characters, removes stopwords and

    filters out all but proper nouns, nounts, verbs and adjectives.

    Parameters
    ----------
    alpha : str

            The input string.

    use_nlp : bool, default False

            Indicates whether Spacy needs to use NLP. Enable this when using this function on its own.

            Should be set to False if used inside nlp.pipeline   

     Returns
    -------
    ' '.join(beta) : a concatenated list of lemmatised tokens, i.e. a processed string

    Notes
    -----
    Fails if alpha is an NA value. Performance decreases as len(alpha) gets large.
    Use together with nlp.pipeline for batch processing.

    """

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])

    if use_nlp:

        alpha = nlp(alpha)

        

    beta = []

    for tok in alpha:

        if all([tok.is_alpha, not tok.is_stop, tok.pos_ in ['PROPN', 'NOUN', 'VERB', 'ADJ']]):

            beta.append(tok.lemma_)

            
    text = ' '.join(beta)
    text = text.lower()
    return text


def tidy(text, http = True, punc = True, lem = False, stop_w = True):
  
    
    if http is True:
        text = re.sub("https?:\/\/t.co\/[A-Za-z0-9]*", '', text)

    # stop words
    if stop_w == True:
        text = [word for word in word_tokenize(text) if not word.lower() in nltk_st]
        text = ' '.join(text)

    elif stop_w == 'spacy':
        text = [text for word in word_tokenize(text) if not word.lower() in spacy_st]
        text = ' '.join(text)

    # lemmitizing
    if lem == True:
        lemmatized = [word.lemma_ for word in sp(text)]
        text = ' '.join(lemmatized)

    # punctuation removal
    if punc is True:
        text = text.translate(str.maketrans('', '', string.punctuation))
        
    # removing extra space
    text = re.sub("\s+", ' ', text)
    
    return text

def remove_stopwords(text, costum_stopwords = False): 
    stop_words = set(stopwords.words("english")) 
    
    if costum_stopwords == True:
        stop_words_custom =  []
    
        newstop_words = stop_words_custom + stop_words
              
        stop_tokens = []
        for i in newstop_words:
            tok = word_tokenize(i) 
            stop_tokens.append(tok)
        y = list(chain(*stop_tokens))
    
        stop_words.update(y)
        
    else:
        stop_words = stop_words
        
    word_tokens = word_tokenize(text) 
    filtered_text = [word for word in word_tokens if word not in stop_words] 
    return ' '.join(filtered_text)


def replace_abb(s):
 
    # replace abbrevations with full information    
    s = re.sub('mett', 'management effectiveness tracking tool', s)
    s = re.sub('nsap', 'national strategic action plan', s)
    s = re.sub('tda', 'transboundary diagnostic analysis', s)
    s = re.sub('ffa', 'pacific islands forum fisheries agency', s)    
    s = re.sub('wcpfc', 'western and central pacific fisheries commission', s)
    s = re.sub('cmm', 'conservation and management measures', s)
    s = re.sub('lsm', 'large scale mining', s)
    s = re.sub(' bd ', 'biodiversity', s) 
    s = re.sub(' ee ', 'energy efficient', s)   
    s = re.sub(' slm ', 'sustainable land management', s)   
    s = re.sub(' sfm ', 'sustainable forest management', s)  
    s = re.sub(' cfr ', 'cape floristic region', s)      
    s = re.sub(' pop ', 'persistent organic pollutants', s)
    s = re.sub(' pops ', 'persistent organic pollutants', s)
    s = re.sub(' upop ', 'unintended persistent organic pollutants', s)
    s = re.sub(' upops ', 'unintended persistent organic pollutants', s)
    s = re.sub(' pcb ', 'polychlorinated biphenyl', s)
    s = re.sub(' pa ', 'protected area', s)
    s = re.sub(' sgp ', 'small grant programme', s)   
    return s


def detect_lang(dataframe, column, lang_to_detect, printing = False, translate = False, remove = False):
    
    pims_id = []
    for index, row in dataframe.iterrows():
        language = detect_langs(row[column])
        
        for each_lang in language:
            if (each_lang.lang == lang_to_detect):
                original_text = row[column]
                
                if translate == True:
                    row[column] = model.translate(row[column], source_lang = lang_to_detect, target_lang = "en")
                else:
                    pass
                
                pims_id.append(row['PIMS_ID'])
                
                if printing == True:
                    print('_________________________')
                    print('PIMS ID:', row['PIMS_ID'])

                    print('')
                    print(original_text)
                    print('')
                    print(row[column])                    
                else:
                    pass
                
    if remove == True:        
        dataframe = dataframe[~dataframe['PIMS_ID'].isin(pims_id)]
    else:
        pass
 
    
    return dataframe


def split_at_length(dataframe, column, length):
    wrapped = []
    for i in dataframe[column]:
        wrapped.append(wrap(i, length))

    dataframe = dataframe.assign(wrapped=wrapped)
    dataframe['wrapped'] = dataframe['wrapped'].apply(lambda x: ', '.join(map(str, x)))

    splitted = pd.concat([pd.Series(row['PIMS_ID'], row['wrapped'].split(", "), )              
                        for _, row in dataframe.iterrows()]).reset_index()

    splitted = splitted.rename(columns={"index": "text", 0: "PIMS_ID"})
    
    return dataframe, splitted