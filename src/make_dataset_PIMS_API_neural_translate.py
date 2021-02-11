# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:28:28 2020

@author: jonas.nothnagel

@title: PIMS_API_make_dataset

@description: set of functions that ingest raw data from PIMS+ API and does minimal processing.
"""

#%%
'''import packages'''
import pandas as pd 
import pickle
import os
import requests

#languages
from langdetect.lang_detect_exception import LangDetectException
from langdetect import detect_langs

from easynmt import EasyNMT
model = EasyNMT('opus-mt')


#%% 
'''Pull live data from API'''
def pull_api():
    url = 'https://IICPSD:PDyq6jRptHnOrS79S9US@api.undpgefpims.org/logframe/v1/project-time-lines'

    r = requests.get(url)
    data = r.json()
    
    '''if we just want to keep individual project IDs:'''
    clean_list = []
    project = []
    for i in data:
        for fname, content in i.items():
            if fname =='project_id' and content not in project:
                project.append(content)
                clean_list.append(i)
    
    return data, clean_list

#%%  
'''transform json to set of lists'''
def data_to_single_lists(data):   


    PIMS_ID = []
    lead_country = []
    description = []
    log_frame = []
    technical_team = []
    focal_area = []
    signature_programmes = []
    full_title = []
    keywords = []
    region = []
    sofPeriod = []
    grant_amount = []    
    is_sids = []
    keyword = []
    project_time_line_key = []
    is_ldc = []
    name = []
    region = []
    subRegion = []
    projectScope = []
    projectStage = []
    projectSubStage = []
    projectStatus = []
    sofFamily = []
    sourcesOfFunds = []
    participatingCountries = []
    projectSectors = []
    resultAreas = []
    jointAgencies = []
        

    for i in data:      
        for fname, content in i.items():
                if fname =='developmentObjectiveOrOutcome':
                    log_frame.append(content)
                content = str(content)
                if fname =='project_id':
                    PIMS_ID.append(content)
                if fname == 'leadCountry':
                    lead_country.append(content)
                if fname =='description':
                    description.append(content)
                if fname == 'technicalTeam':
                    technical_team.append(content)
                if fname =='focalArea':
                    focal_area.append(content)
                if fname =='signatureProgrammes':
                    signature_programmes.append(content)
                if fname =='full_title':
                    full_title.append(content)
                if fname =='keyword':
                    keywords.append(content)
                if fname =='region':
                    region.append(content)
                if fname =='sofPeriod':
                    sofPeriod.append(content)
                if fname =='grant_amount':
                    grant_amount.append(content)  
                if fname =='is_sids':
                    is_sids.append(content)
                if fname == 'keyword':
                    keyword.append(content)
                if fname =='project_time_line_key':
                    project_time_line_key.append(content)
                if fname == 'is_ldc':
                    is_ldc.append(content)
                if fname =='name':
                    name.append(content)
                if fname =='subRegion':
                    subRegion.append(content)
                if fname =='projectScope':
                    projectScope.append(content)
                if fname =='projectStage':
                    projectStage.append(content)
                if fname =='projectSubStage':
                    projectSubStage.append(content)
                if fname =='projectStatus':
                    projectStatus.append(content)
                if fname =='sofFamily':
                    sofFamily.append(content)  
                if fname =='sourcesOfFunds':
                    sourcesOfFunds.append(content)
                if fname =='participatingCountries':
                    participatingCountries.append(content)
                if fname =='projectSectors':
                    projectSectors.append(content)
                if fname =='resultAreas':
                    resultAreas.append(content)
                if fname =='jointAgencies':
                    jointAgencies.append(content)  

    '''convert grant_amount to integers '''
    
    grant_amount_int = []
    for i in grant_amount:
    
        i = str(i) 
    
        i = i.split(',')[0]
    
        i = i.replace('.', '')
    
        if i != "nan":
    
            i = int(i)
            
        else:
            i == ""
        
        grant_amount_int.append(i)  
                
    return PIMS_ID, lead_country, description, log_frame, technical_team, focal_area, signature_programmes, \
        full_title, keywords, region, sofPeriod, grant_amount_int, is_sids, keyword, project_time_line_key, \
        is_ldc, name, region, subRegion, projectScope, projectStage, projectSubStage, projectStatus, sofFamily, \
        sourcesOfFunds, participatingCountries, projectSectors, resultAreas, jointAgencies


#%%   

def parse_log_frame(data, translate = False):
    
    model = EasyNMT('opus-mt')
    '''
    Logframe parser: 
    1. Parses log-frames into objectives, outcomes and indicators
    2. Merges relevant fields together to obtain full text
    2. Only parses non-empty frames.
    3. replaces None types and empty fields with spaces
    4. Detects non english chunks in log-frames and translates them to English.
    '''
    if translate == True:
        print("non english text will be detected and translated...takes a long time.")
    else:
        pass
    '''create empty list to append relevant fields'''
    objective_indicators = []
    objective_content = []
    outcome_indicators = []
    outcome_content = []
    types = []
    
    for i in data:
        clean_accumulated_indicators = []
        content = []
        b_clean_accumulated_indicators = []
        b_content = []    
        '''look at non-empty log_frames:'''
        if i != []:
            
            '''first level: the dictionaries in the list. Here we have 7:'''
            for dic in i:
                
                '''the dictionaries have two types: objective or outcomes:'''
                for key, text in dic.items():
                    if key == "type":
                        
                        '''first generate list that appends all objectives in it:'''
                        types.append(text)
                        if "objective" in types:
                            
                            if text == "objective":
                                
                                '''access the items in the objectives:'''
                                for obj, text in dic.items():
        
                                    'access content of objective items:'
                                    if obj =="content":

                                        
                                        '''detect language and remove non english posts'''
                                        if translate == True:
                                            try:
                                                language = detect_langs(text)
                                                for each_lang in language:
                                                    if each_lang.lang == "fr" and each_lang.prob > 0.75:
                                                        print(text)
                                                        translated_text = model.translate(text, source_lang = "fr", target_lang='en')
                                                        print(translated_text)
                                                        content.append(translated_text)
                                                    if each_lang.lang == "es" and each_lang.prob > 0.75:
                                                        print(text)
                                                        translated_text = model.translate(text, source_lang = "es", target_lang='en')
                                                        print(translated_text)
                                                        content.append(translated_text)                                                     
                                                    else:
                                                        content.append(text)  
                                            except LangDetectException as err:
                                                content.append(text)
                                                                    
                                        else:
                                            content.append(text)
                                            
                                    '''access indicator items in the objectives:'''
                                    if obj =="indicator":
                                        
                                        #clean_accumulated_indicators = []
                                        for indicator in text:
        
                                            '''access the dictionaries items within the indicator dictionary:'''
                                            for x, text in indicator.items():
                                                
                                                if x == "content":
                                                    if text != None:
                                                        if translate == True:
                                                        
                                                            try:
                                                                language = detect_langs(text)
                                                                for each_lang in language:
                                                                    if each_lang.lang == "fr" and each_lang.prob > 0.75:
                                                                        print(text)
                                                                        translated_text = model.translate(text, source_lang = "fr", target_lang='en')
                                                                        print(translated_text)
                                                                        clean_accumulated_indicators.append(translated_text)
                                                                    if each_lang.lang == "es" and each_lang.prob > 0.75:
                                                                        print(text)
                                                                        translated_text = model.translate(text, source_lang = "es", target_lang='en')
                                                                        print(translated_text)
                                                                        clean_accumulated_indicators.append(translated_text)                                                      
                                                                    else:
                                                                        clean_accumulated_indicators.append(text)  
                                                            except LangDetectException as err:
                                                                clean_accumulated_indicators.append(text)
                                                        else:
                                                            clean_accumulated_indicators.append(text)
                                                                                                                        
                                                    else:
                                                        text == ''
                                                        clean_accumulated_indicators.append('nan') 
                                                if x == "end_of_project_target_level":
                                                    if text != None:
                                                        
                                                        if translate == True:
                                                            try:
                                                                language = detect_langs(text)
                                                                for each_lang in language:
                                                                    if each_lang.lang == "fr" and each_lang.prob > 0.75:
                                                                        print(text)
                                                                        translated_text = model.translate(text, source_lang = "fr", target_lang='en')
                                                                        print(translated_text)
                                                                        clean_accumulated_indicators.append(translated_text)
                                                                    if each_lang.lang == "es" and each_lang.prob > 0.75:
                                                                        print(text)
                                                                        translated_text = model.translate(text, source_lang = "es", target_lang='en')
                                                                        print(translated_text)
                                                                        clean_accumulated_indicators.append(translated_text)                                                      
                                                                    else:
                                                                        clean_accumulated_indicators.append(text)  
                                                            except LangDetectException as err:
                                                                clean_accumulated_indicators.append(text)
                                                        else:
                                                            clean_accumulated_indicators.append(text)
                                                            
                                                    else:
                                                        text == ''
                                                        clean_accumulated_indicators.append('nan')
                                                if x == "mid_term_target_level":
                                                    if text != None:
                                                        
                                                        if translate == True:
                                                            try:
                                                                language = detect_langs(text)
                                                                for each_lang in language:
                                                                    if each_lang.lang == "fr" and each_lang.prob > 0.75:
                                                                        print(text)
                                                                        translated_text = model.translate(text, source_lang = "fr", target_lang='en')
                                                                        print(translated_text)
                                                                        clean_accumulated_indicators.append(translated_text)
                                                                    if each_lang.lang == "es" and each_lang.prob > 0.75:
                                                                        print(text)
                                                                        translated_text = model.translate(text, source_lang = "es", target_lang='en')
                                                                        print(translated_text)
                                                                        clean_accumulated_indicators.append(translated_text)                                                     
                                                                    else:
                                                                        clean_accumulated_indicators.append(text)  
                                                            except LangDetectException as err:
                                                                clean_accumulated_indicators.append(text)
                                                        else:
                                                            clean_accumulated_indicators.append(text)
                                                            
                                                    else:
                                                        text == ''
                                                        clean_accumulated_indicators.append('nan')
                                                if x == "baseline_level":
                                                    if text != None:
                                                        
                                                        if translate == True:

                                                            try:
                                                                language = detect_langs(text)
                                                                for each_lang in language:
                                                                    if each_lang.lang == "fr" and each_lang.prob > 0.75:
                                                                        print(text)
                                                                        translated_text = model.translate(text, source_lang = "fr", target_lang='en')
                                                                        print(translated_text)
                                                                        clean_accumulated_indicators.append(translated_text)
                                                                    if each_lang.lang == "es" and each_lang.prob > 0.75:
                                                                        print(text)
                                                                        translated_text = model.translate(text, source_lang = "es", target_lang='en')
                                                                        print(translated_text)
                                                                        clean_accumulated_indicators.append(translated_text)                                                  
                                                                    else:
                                                                        clean_accumulated_indicators.append(text)  
                                                            except LangDetectException as err:
                                                                clean_accumulated_indicators.append(text)
                                                            else:
                                                                clean_accumulated_indicators.append(text)
                                                                
                                                    else:
                                                        text == ''
                                                        clean_accumulated_indicators.append('nan')
                                            
                                                '''fifth and sixth level:'''
                                                # append only contents with entries - no timestamp here:
                                                if x == "datedLevel":
                                                    for i in text:
                                                        for y, z in i.items():
                                                            if y == "content":
                                                                if z != None:
                                                                    if translate == True:
                                                                        try:
                                                                            language = detect_langs(z)
                                                                            for each_lang in language:
                                                                               
                                                                                if each_lang.lang == "fr" and each_lang.prob > 0.75:
                                                                                    print(text)
                                                                                    translated_text = model.translate(z, source_lang ="fr", target_lang='en')
                                                                                    print(translated_text)
                                                                                    clean_accumulated_indicators.append(translated_text)
                                                                                if each_lang.lang == "es" and each_lang.prob > 0.75:
                                                                                    print(text)
                                                                                    translated_text = model.translate(z, source_lang = "es", target_lang="en")
                                                                                    print(translated_text)
                                                                                    clean_accumulated_indicators.append(translated_text)
                                                                                else:
                                                                                    clean_accumulated_indicators.append(z)  
                                                                        except LangDetectException as err:
                                                                            clean_accumulated_indicators.append(z)
                                                                    else:
                                                                        clean_accumulated_indicators.append(z)
    
                        if "outcome" in types:

                            if text == "outcome":
                                    
                                '''access the items in the objectives:'''
                                for obj, text in dic.items():
                            
                                    'access content of objective items:'
                                    if obj =="content":
                                        
                                        if translate == True:
                                            try:
                                                language = detect_langs(text)
                                                for each_lang in language:
                                                    
                                                    if each_lang.lang == "fr" and each_lang.prob > 0.75:
                                                        print(text)
                                                        translated_text = model.translate(text, source_lang = "fr", target_lang='en')
                                                        print(translated_text)
                                                        b_content.append(translated_text)
                                                    if each_lang.lang == "es" and each_lang.prob > 0.75:
                                                        print(text)
                                                        translated_text = model.translate(text, source_lang = "es", target_lang='en')
                                                        print(translated_text)
                                                        b_content.append(translated_text)                                                      
                                                    else:
                                                        b_content.append(text)  
                                            except LangDetectException as err:
                                                b_content.append(text)
                                        else:
                                            b_content.append(text)
                                            
                                        
                                    '''access indicator items in the objectives:'''
                                    if obj =="indicator":
                                        for indicator in text:
        
                                            '''access the dictionaries items within the indicator dictionary:'''
                                            for x, text in indicator.items():
                                                
                                                if x == "content":
                                                    if text != None:
                                                        
                                                        if translate == True:
                                                            try:
                                                                language = detect_langs(text)
                                                                for each_lang in language:
                                                                    if each_lang.lang == "fr" and each_lang.prob > 0.75:
                                                                        print(text)
                                                                        translated_text = model.translate(text, source_lang = "fr", target_lang='en')
                                                                        print(translated_text)
                                                                        b_clean_accumulated_indicators.append(translated_text)
                                                                    if each_lang.lang == "es" and each_lang.prob > 0.75:
                                                                        print(text)
                                                                        translated_text = model.translate(text, source_lang = "es", target_lang='en')
                                                                        print(translated_text)
                                                                        b_clean_accumulated_indicators.append(translated_text)                                                       
                                                                    else:
                                                                        b_clean_accumulated_indicators.append(text)  
                                                            except LangDetectException as err:
                                                                b_clean_accumulated_indicators.append(text)
                                                        else:
                                                            b_clean_accumulated_indicators.append(text)
                                                            
                                                    else:
                                                        text == ''
                                                        b_clean_accumulated_indicators.append('nan') 
                                                if x == "end_of_project_target_level":
                                                    if text != None:
                                                        
                                                        if translate == True:
                                                            try:
                                                                language = detect_langs(text)
                                                                for each_lang in language:
                                                                    if each_lang.lang == "fr" and each_lang.prob > 0.75:
                                                                        print(text)
                                                                        translated_text = model.translate(text, source_lang = "fr", target_lang='en')
                                                                        print(translated_text)
                                                                        b_clean_accumulated_indicators.append(translated_text)
                                                                    if each_lang.lang == "es" and each_lang.prob > 0.75:
                                                                        print(text)
                                                                        translated_text = model.translate(text, source_lang = "es", target_lang='en')
                                                                        print(translated_text)
                                                                        b_clean_accumulated_indicators.append(translated_text)                                                     
                                                                    else:
                                                                        b_clean_accumulated_indicators.append(text)  
                                                            except LangDetectException as err:
                                                                b_clean_accumulated_indicators.append(text)
                                                        else:
                                                            b_clean_accumulated_indicators.append(text)
                                                            
                                                    else:
                                                        text == ''
                                                        b_clean_accumulated_indicators.append('nan')
                                                        
                                                if x == "mid_term_target_level":
                                                    if text != None:
                                                        
                                                        if translate == True: 
                                                            try:
                                                                language = detect_langs(text)
                                                                for each_lang in language:
                                                                    if each_lang.lang == "fr" and each_lang.prob > 0.75:
                                                                        print(text)
                                                                        translated_text = model.translate(text, source_lang = "fr", target_lang='en')
                                                                        print(translated_text)
                                                                        b_clean_accumulated_indicators.append(translated_text)
                                                                    if each_lang.lang == "es" and each_lang.prob > 0.75:
                                                                        print(text)
                                                                        translated_text = model.translate(text, source_lang = "es", target_lang='en')
                                                                        print(translated_text)
                                                                        b_clean_accumulated_indicators.append(translated_text)                                                     
                                                                    else:
                                                                        b_clean_accumulated_indicators.append(text)  
                                                            except LangDetectException as err:
                                                                b_clean_accumulated_indicators.append(text)
                                                        else:
                                                            b_clean_accumulated_indicators.append(text)
                                                            
                                                    else:
                                                        text == ''
                                                        b_clean_accumulated_indicators.append('nan')
                                                        
                                                if x == "baseline_level":
                                                    
                                                    if text != None:
                                                        
                                                        if translate == True:

                                                            try:
                                                                language = detect_langs(text)
                                                                for each_lang in language:
                                                                    
                                                                    if each_lang.lang == "fr" and each_lang.prob > 0.75:
                                                                        print(text)
                                                                        translated_text = model.translate(text, source_lang = "fr", target_lang='en')
                                                                        print(translated_text)
                                                                        b_clean_accumulated_indicators.append(translated_text)
                                                                    if each_lang.lang == "es" and each_lang.prob > 0.75:
                                                                        print(text)
                                                                        translated_text = model.translate(text, source_lang = "es", target_lang='en')
                                                                        print(translated_text)
                                                                        b_clean_accumulated_indicators.append(translated_text)                                                      
                                                                    else:
                                                                        b_clean_accumulated_indicators.append(text)  
                                                            except LangDetectException as err:
                                                                b_clean_accumulated_indicators.append(text)
                                                        else:
                                                            b_clean_accumulated_indicators.append(text)
                                                            
                                                    else:
                                                        text == ''
                                                        b_clean_accumulated_indicators.append('nan')
                                            
                                                '''fifth and sixth level:'''
                                                # append only contents with entries - no timestamp here:
                                                if x == "datedLevel":
                                                    for i in text:
                                                        for y, z in i.items():
                                                            if y == "content":
                                                                
                                                                if z != None:
                                                                    
                                                                    if translate == True:
                                                                        try:
                                                                            language = detect_langs(z)
                                                                            for each_lang in language:                                                                            
                                                                                if each_lang.lang == "fr" and each_lang.prob > 0.75:
                                                                                    print(text)
                                                                                    translated_text = model.translate(z, source_lang ="fr", target_lang='en')
                                                                                    print(translated_text)
                                                                                    b_clean_accumulated_indicators.append(translated_text)
                                                                                if each_lang.lang == "es" and each_lang.prob > 0.75:
                                                                                    print(text)
                                                                                    translated_text = model.translate(z, source_lang = "es", target_lang="en")
                                                                                    print(translated_text)
                                                                                    b_clean_accumulated_indicators.append(translated_text)
                                                                                else:
                                                                                    clean_accumulated_indicators.append(z)  
                                                                        except LangDetectException as err:
                                                                            b_clean_accumulated_indicators.append(z)
                                                                    else:
                                                                        b_clean_accumulated_indicators.append(z)
                            
            objective_indicators.append(clean_accumulated_indicators)
            objective_content.append(content)
                            
            outcome_indicators.append(b_clean_accumulated_indicators)
            outcome_content.append(b_content)
    
        else:
            objective_indicators.append('')
            objective_content.append('')
            
            outcome_indicators.append('')
            outcome_content.append('')
            
    
    objective_tuple = zip(objective_indicators, objective_content)     
    df_1 = pd.DataFrame(objective_tuple, columns = ['a', 'b'])   
    df_1['a'] = df_1['a'].astype(str)
    df_1['b'] = df_1['b'].astype(str)
    df_1['extract'] = df_1['a']  + ' ' +  df_1['b']
    objectives = df_1['extract'].to_list()
    
    outcome_tuple = zip(outcome_indicators, outcome_content)     
    df_2 = pd.DataFrame(outcome_tuple, columns = ['a', 'b'])   
    df_2['a'] = df_2['a'].astype(str)
    df_2['b'] = df_2['b'].astype(str)
    df_2['extract'] = df_2['a']  + ' ' + df_2['b']
    outcomes = df_2['extract'].to_list()
        
    full_tuple = zip(objectives, outcomes)     
    df_3 = pd.DataFrame(full_tuple, columns = ['a', 'b'])   
    df_3['a'] = df_3['a'].astype(str)
    df_3['b'] = df_3['b'].astype(str)
    df_3['extract'] = df_3['a']  + ' ' + df_3['b']
    full_obj_or_outcome = df_3['extract'].to_list()
        
    print(len(objectives))
    print(len(outcomes))
    print(len(full_obj_or_outcome))

    
    return objectives, outcomes, full_obj_or_outcome

def clean_pims_data(df_logs):
    
    df_logs = df_logs[['PIMS_ID', 'pims_description', 'full_obj_or_outcome', 'pims_objectives', 'pims_outcomes']]
    df_logs.full_obj_or_outcome = df_logs.full_obj_or_outcome.str.replace('[', ' ').str.replace(']', ' ').str.replace(',', ' ').str.replace("'", '').str.replace("   ", '')
    
    df_logs['PIMS_ID'] = df_logs['PIMS_ID'].astype(int)

    #for index, row in df_logs.iterrows():
    
        # if we only want to keep a certain minimum length:
        
        #if len(row['pims_description']) < 150:
            #df_logs.at[index, 'pims_description'] = ''
        
        #if len(row['full_obj_or_outcome']) < 150:
            #df_logs.at[index, 'full_obj_or_outcome'] = ''
    
        #if row['pims_description'] == '' and row['full_obj_or_outcome'] == '':
            #df_logs.drop(index, inplace=True)
        
    #df_logs = df_logs.reset_index(drop=True)
    
    df_logs['pims_text'] = df_logs['full_obj_or_outcome']
    
    for index, row in df_logs.iterrows():
            
        if row['pims_description'] == '' and row['full_obj_or_outcome'] != '':
            df_logs.at[index, 'pims_text'] = row['full_obj_or_outcome']
            
        if row['pims_description'] != '' and row['full_obj_or_outcome'] == '':
            df_logs.at[index, 'pims_text'] = row['pims_description']
    
    filter = df_logs["pims_text"] != ""
    logs = df_logs[filter]
    
    #remove hyphenations, numbers and whitespaces and remaining trailing characters
    logs = logs.replace('(?<=[a-z])-(?=[a-z])', '',regex=True)
#     logs = logs.replace(r'\n', ' ',regex=True)
#     logs = logs.replace(r'\r', ' ',regex=True)    
#     logs = logs.replace(r'\d+', ' ',regex=True)
#     logs = logs.replace(r'\W+', ' ',regex=True)
#     logs.pims_text = logs.pims_text.str.replace('xa ', ' ')
#     logs.pims_text = logs.pims_text.str.replace('n r n r', ' ')
#     logs.pims_text = logs.pims_text.str.replace(' r n ', ' ')   
#     logs.pims_text = logs.pims_text.str.replace(' r ', ' ')   
#     logs.pims_text = logs.pims_text.str.replace(' n ', ' ')   
#     logs.pims_text = logs.pims_text.str.replace(' r t', ' ')   
#     logs.pims_text = logs.pims_text.str.replace(' n t', ' ')   
    
    return logs

