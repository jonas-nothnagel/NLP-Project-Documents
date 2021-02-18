"""
Created on Fri Oct  9 
@author: Oumaima
@title: clean_dataset
@descriptions: set of functions that enable different level of data cleaning.
"""
#%%
import pandas as pd
import string  
import nltk 
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import re
import nltk
from nltk.tokenize import word_tokenize
from language_detector import detect_language
import pkg_resources
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
if sym_spell.word_count:
    pass
else:
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
#%%

# text level preprocess 
# lowercase + base filter
# some basic normalization
def f_base(s):
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

    return s.strip()


# language detection
def f_lan(s):
    """
    :param s: string to be processed
    :return: boolean (s is English)
    """
    # some reviews are actually english but biased toward french
    return detect_language(s) in {'English', 'French'}



# word level preprocess
# filtering out punctuations and numbers
def f_punct(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with punct and number filter out
    """
    return [word for word in w_list if word.isalpha()]


# selecting nouns
def f_noun(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with only nouns selected
    """
    return [word for (word, pos) in nltk.pos_tag(w_list) if pos[:2] == 'NN']


# typo correction
def f_typo(w_list):
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


def f_stem(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with stemming
    """
    return [p_stemmer.stem(word) for word in w_list]


# filtering out stop words
# create English stop words list
# We can add custom stopwords list later

en_stop = get_stop_words('en')
def f_stopw(w_list):
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
    s = f_base(rw)
    if not f_lan(s):
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
    w_list = f_punct(w_list)
    w_list = f_noun(w_list)
    w_list = f_typo(w_list)
    w_list = f_stem(w_list)
    w_list = f_stopw(w_list)

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
            w_list = f_stem(w_list)
            s = ' '.join(w_list)

        if stopwords:
            w_list = word_tokenize(s)
            w_list = f_stopw(w_list)
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
