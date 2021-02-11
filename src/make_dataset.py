#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:06:19 2020

@author: jonas & oumaima

@tile: make_dataset

@description: script to transform taxonomy from excel sheet to machine readable format in python.
"""
#%%
'''import packages'''
import os

import pandas as pd 
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import re


'''own functions'''
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
import src.clean_dataset as cl
import src.make_dataset_PIMS_API as api 
import src.make_dataset_PIMS_API_neural_translate as api_neural


import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))
from easynmt import EasyNMT
model = EasyNMT('opus-mt')
#%%

def import_raw_data(tagging_table):
    
    '''
    import and minimal processing of taxonomy from excel    
    '''
    
    taxonomy = pd.read_excel(os.path.abspath(os.path.join('..', 'data/raw'))+'/'+tagging_table+'.xlsx')
    # get column names:
    columnNames = taxonomy.iloc[0] 
    taxonomy = taxonomy[1:] 
    taxonomy.columns = columnNames
    print('raw data shape:', taxonomy.shape)
    # delete entries without PIMS ID:
    taxonomy = taxonomy[taxonomy['PIMS #'].notna()]      
    print('only rows that have valid PIMS ID:', taxonomy.shape)                
    # delete columns without names:
    taxonomy = taxonomy.loc[:, taxonomy.columns.notnull()]   
    print('only columns with entries:', taxonomy.shape)                
    # remove white spaces in column names and lowercase names:
    taxonomy.columns = taxonomy.columns.str.replace(' ', '_').str.lower()
    # rename pims id column:
    taxonomy = taxonomy.rename(columns={"pims_#": "PIMS_ID"})
    
    return taxonomy

def import_indicators():
    
    indicators = pd.read_excel(os.path.abspath(os.path.join('..', 'data/raw'))+'/indicators_v2.xlsx')
    indicators = indicators.rename(columns={"PIMS ID": "PIMS_ID"})
    indicators.PIMS_ID.fillna(method='ffill', inplace = True)
    indicators = indicators.fillna('')
    indicators = indicators.groupby('PIMS_ID').agg(' '.join)
    indicators = indicators.reset_index(drop=False)
    indicators = indicators[['PIMS_ID', 'Indicator']]

    return indicators
    
    
def import_api_data(all_timelines = True):
    
    '''
    function that imports data from PIMS+ API. 
    '''
    print('downloading data from PIMS+ API....')
    data, clean_list = api_neural.pull_api()
    
    if all_timelines == True:
        PIMS_ID, lead_country, description, log_frame, technical_team, focal_area, signature_programmes, \
             full_title, keywords, region, sofPeriod, grant_amount_int, \
            is_sids, keyword, project_time_line_key, is_ldc, name, region, subRegion, \
            projectScope, projectStage, projectSubStage, projectStatus, sofFamily, \
            sourcesOfFunds, participatingCountries, projectSectors, resultAreas, jointAgencies \
            = api_neural.data_to_single_lists(data) #put in clean_list to only use individual project_IDs.
    else:
        PIMS_ID, lead_country, description, log_frame, technical_team, focal_area, signature_programmes, \
             full_title, keywords, region, sofPeriod, grant_amount_int, \
            is_sids, keyword, project_time_line_key, is_ldc, name, region, subRegion, \
            projectScope, projectStage, projectSubStage, projectStatus, sofFamily, \
            sourcesOfFunds, participatingCountries, projectSectors, resultAreas, jointAgencies \
            = api_neural.data_to_single_lists(clean_list) #put in clean_list to only use individual project_IDs.
        
        
    pims_objectives, pims_outcomes, full_obj_or_outcome = api_neural.parse_log_frame(log_frame, translate = True)

    '''create pandas dataframe and save relevant data:'''
    data_tuple = zip(PIMS_ID, lead_country, description, log_frame, technical_team, focal_area, signature_programmes, 
                     full_title, keywords, region, sofPeriod, grant_amount_int, is_sids, keyword, project_time_line_key,
                     is_ldc, name, subRegion, projectScope, projectStage, projectSubStage, projectStatus, sofFamily, 
                     sourcesOfFunds, participatingCountries, projectSectors, resultAreas, jointAgencies, pims_objectives, pims_outcomes, full_obj_or_outcome)
    
    df_api = pd.DataFrame(data_tuple, columns = ['PIMS_ID', 'lead_country', 'pims_description', 'log_frame', 'technical_team', 'focal_area', 'signature_programmes', 
                     'full_title', 'keywords', 'region', 'sofPeriod', 'grant_amount_int', 'is_sids', 'keyword', 'project_time_line_key',
                     'is_ldc', 'name', 'subRegion', 'projectScope', 'projectStage', 'projectSubStage', 'projectStatus', 'sofFamily', 
                     'sourcesOfFunds', 'participatingCountries', 'projectSectors', 'resultAreas', 'jointAgencies', 'pims_objectives', 'pims_outcomes', 'full_obj_or_outcome'])
    print(len(df_api))
    
    '''Delete all projects with empty logframes and empty descriptions'''
    df_logs = df_api
    for index, row in df_logs.iterrows():
    
        if row['pims_description'] == '' and row['full_obj_or_outcome'] == '   ':
            df_logs.drop(index, inplace=True)

    df_logs = df_logs.reset_index(drop=True)
    print('after deleting empty logframes:', len(df_logs))
    
    #print('pickling data....')
    #with open(os.path.abspath(os.path.join('..', 'data/interim'))+'/pims_logframes.pkl', 'wb') as handle:
        #pickle.dump(df_logs, handle, protocol=pickle.HIGHEST_PROTOCOL)   
    print('saving data...')
    
    print('done!')
    df_logs.to_excel(os.path.abspath(os.path.join('..', 'data/interim'))+'/pims_logframes.xlsx', index=False)    
    return df_api, df_logs

        
def create_training_texts(dataframe, compare_with_API = False, under_implementation = True, add_indicators = True, replace_short = True):
    
    """
    import_taxonomy, import_indicators and import_api_data have to be run before to allow full functionality:
    """
    
    """
    1. Append indicators if possible:
    """
    if add_indicators == True:
        print('appending indicators...')
        dataframe = dataframe.merge(import_indicators(), how='left', left_on='PIMS_ID', right_on='PIMS_ID')
        dataframe['Indicator'] = dataframe['Indicator'].fillna('') 
        print('done!')
        print('______________________________')
        print('')  
    else:
        print('not appending indicators')
        pass
    
    
    
    """
    2. Takes in whole taxonomy and outputs different training data text fields and replaces "nan" with empty spaces. 
    """
    
    # objectives
    dataframe['objectives'] = dataframe['project_objective'].fillna('').astype(str) + dataframe['project_objective_2'].fillna('').astype(str)
   
    # rename description
    dataframe['description'] = dataframe['project_description'].fillna('').astype(str)

    
    # outcomes
    dataframe['outcomes'] = dataframe['outcome_1'].fillna('').astype(str)
    
    # outputs
    dataframe['outputs'] = dataframe[['output_1.1', 'output_1.2', 'output_1.3',
                            'output_1.4', 'output_1.5', 'outcome_2', 'output_2.1', 'output_2.2',
                            'output_2.3', 'output_2.4', 'output_2.5', 'outcome_3', 'output_3.1',
                            'output_3.2', 'output_3.3', 'output_3.4', 'output_3.5', 'outcome_4',
                            'output_4.1', 'output_4.2', 'output_4.3', 'output_4.4', 'output_4.5', 
                            'outcome_5', 'output_5.1', 'output_5.2', 'output_5.3',
                            'output_5.4_(no_entry)', 'output_5.5_(no_entry)',
                            'outcome_6_(no_entry)', 'output_6.1', 'output_6.2']].fillna('').astype(str).agg(' '.join, axis=1)

  
    #  'output_6.3', 'output_6.4_(no_entry)', 'output_6.5_(no_entry)','outcome_7_(no_entry)', 'output_7.1', 'output_7.2_(no_entry)','output_7.3_(no_entry)', 'output_7.4_(no_entry)','output_7.5_(no_entry)' #tagging_table 1 has more outputs
    
    
    dataframe['logframe'] = dataframe[['objectives', 'outcomes', 'outputs']].agg(' '.join, axis=1)
    
    dataframe['log_des'] = dataframe['description'] + dataframe['logframe'] 
    
    dataframe['all_text_and_title'] = dataframe['log_des'] + dataframe['title']
    
    dataframe['all_text'] = dataframe['description'] + dataframe['logframe'] + dataframe['Indicator']
    
    print('extracting and merging all text fields - descriptions, objectives, outcomes, outputs, indicators...')
    print('done!')
    print('______________________________')
    print('')  
    
    """
    3. Keep only projects that are currently under implementation:
    """
    
    if under_implementation == True:
        print(dataframe.shape)
        print('keep only projects that are currently under implementation...')
        dataframe = dataframe[dataframe['project_status'].isin(['Under Implementation0825'])]
        print(dataframe.shape)
        dataframe = dataframe.reset_index(drop=True)        
        print('done!')
        print('______________________________')
        print('')  
        
    else:
        print('keep all projects regardless of project status')
        dataframe = dataframe        
        
    """    
    4. if bool is set as True: Compare with downloaded logframes, descriptions and objectives from PIMS+ to see if they 
    match in length and compleetness.
    - Replace empty fields with non-empty fiels if applicable.
    """
    
    if compare_with_API == True:
        '''compare with PIMS+ projects/logframes etc and only keep most relevant'''
        
        print('Pims_plus API is considered to complete training data....')
                           
        #with open(os.path.abspath(os.path.join('..', 'data/interim'))+'/pims_logframes.pkl', 'rb') as handle:
            #df_logs = pickle.load(handle)
            
        df_logs = pd.read_excel(os.path.abspath(os.path.join('..', 'data/interim'))+'/pims_logframes.xlsx')    
        
        #clean api data for processing:
        pims = api.clean_pims_data(df_logs)
        dataframe = dataframe.merge(pims, how='left', left_on='PIMS_ID', right_on='PIMS_ID')
        
        #convert to string and replaces nan:
        dataframe['all_text'] = dataframe['all_text'].str.replace('nan', '')
        dataframe['pims_text'] = dataframe['pims_text'].astype(str)
        dataframe['pims_text'] = dataframe['pims_text'].str.replace('nan', '')
        
        for index, row in dataframe.iterrows():
    
            if len(row['all_text']) < 50:
                dataframe.at[index, 'all_text'] = ''
                
            #give more priority to text from taxonomy and only replace with PIMS API text if empty or under 600:
            if row['all_text'] == '' and row['pims_text'] != '':
                dataframe.at[index, 'all_text'] = row['pims_text']
            
            if replace_short == True:    
                if len(row['all_text']) < 600 and len(row['pims_text']) > 600:
                    print('replaced short with long!')
                    dataframe.at[index, 'all_text'] = row['pims_text']
            else:
                pass
        """
        Remove exact duplicates (each column entry is identical
        """
        print('Remove duplications from different time lines - delete exact otherwise keep latest...')
        print('done!')
        dataframe.sort_values(by=['PIMS_ID'], inplace=True)
        dataframe = pd.DataFrame.drop_duplicates(dataframe)        
        
        """
        Remove PIMS_ID duplicates and keep last entry which represents last timeline entry - only few 
        so should be fine. Change this if we want to consider more timelines or merge the timelines together.
        Check if timesline are almost identical first! This is the case here - so I just keep the last since they
        build often up on each other.
        """

        dataframe  = pd.DataFrame.drop_duplicates(dataframe, subset=['PIMS_ID'], keep = 'first')
        print('shape of data after removing timeline duplications:')
        print(len(dataframe))        
        print('______________________________')
        print('')          
        
               
    else:
        print('only taxonomy data is considered')
        pims = []
        pass
    
    
    """
    5. Check if project contain any text or only titles and are thus evaluated with project documents.
        - perhapos add: if yes: scrape project document and append logframe from it to data.
    """

    print('flag projects that were tagged by looking at project document...')
    title = dataframe[['PIMS_ID', 'title', 'all_text_and_title']]
    title.all_text_and_title = title.all_text_and_title.str.replace(' ', '')
    title.title = title.title.str.replace(' ', '')
    only_title = title.loc[title['all_text_and_title'] == title['title']]

    dataframe = dataframe.merge(only_title, indicator=True, how='left', left_on='PIMS_ID', right_on='PIMS_ID').rename(columns={'_merge':'title_only'})
    dataframe.title_only = dataframe.title_only.replace('left_only', False)
    dataframe.title_only = dataframe.title_only.replace('both', True)
    print('number of documents with and without proper text and only titles', dataframe['title_only'].value_counts())

    # rename merged columns
    dataframe = dataframe.rename(columns={"description_x": "description"})   
    dataframe = dataframe.rename(columns={"title_x": "title"})   
    dataframe = dataframe.rename(columns={"all_text_and_title_x": "all_text_and_title"})   
    print('done!')
    print('______________________________')
    print('')            
       
    dataframe = dataframe.reset_index(drop=True)        
    
    
    """
    6. Compare with list of not yet implemented and cancelled project and remove if True.
    """
    print('Checking list of PIMS_ID that are not implemented or cancelled and remove from dataframa if no text exists...')
    not_imp_cancel = [5753, 5697, 6005, 5886, 5854, 5741, 5728, 6266, 6172, 6117,6116,6100,
                6044,6043,5595,5434,6434,6422,6421,6345,6344,6332,6272,6212,6209,6250,
                5986,2796,6152,6140,6055,6041,5275,5863,6151]
    

    dataframe['all_text'] = dataframe['all_text'].fillna('').astype(str)
    i = 0 
    for index, row in dataframe.iterrows():

        #only relevant if project is existing / not empty of course:
        if row['PIMS_ID'] in not_imp_cancel:
            if row['all_text'] == '':
                i = i +1

                dataframe.drop(index, inplace=True)
        
    print('Number of projects that will be removed:', i)
    print('______________________________')
    print('')            
    """
    7. Drop unneeded columns:
    """
    
    dataframe = dataframe.drop(columns=['title_y', 'all_text_and_title', 'all_text_and_title_y', 'output_1.1', 'output_1.2', 'output_1.3',
                            'output_1.4', 'output_1.5', 'outcome_2', 'output_2.1', 'output_2.2',
                            'output_2.3', 'output_2.4', 'output_2.5', 'outcome_3', 'output_3.1',
                            'output_3.2', 'output_3.3', 'output_3.4', 'output_3.5', 'outcome_4',
                            'output_4.1', 'output_4.2', 'output_4.3', 'output_4.4', 'output_4.5', 
                            'outcome_5', 'output_5.1', 'output_5.2', 'output_5.3',
                            'output_5.4_(no_entry)', 'output_5.5_(no_entry)',
                            'outcome_6_(no_entry)', 'output_6.1', 'output_6.2',
                            'outcome_1', 'rta','project_objective', 'project_objective_2', 'project_description', 'taggers'])

    # 'output_6.3','output_6.4_(no_entry)', 'output_6.5_(no_entry)', 'outcome_7_(no_entry)', 'output_7.1', 'output_7.2_(no_entry), 'output_7.3_(no_entry)', 'output_7.4_(no_entry)','output_7.5_(no_entry)' #tagging_table has less outputs.
    

    
    
    
    print('processing done!')
    print('final shape of data:', dataframe.shape)
    return dataframe, pims
    
def create_subset(dataframe, column_title):
    
    '''
    Takes datafram as input and column name and outputs a dataframe with two columns: project_id and column without empty fields.
        - can also be appended with more meta_data than only project_id for downstream tasks.    
    '''
    
    print('deleting all empty fields and creating subset for:', column_title)   
    #keep only projects with column_title that contain alphabetic letter:
    
    print('___________')
    print('original data size is:', len(dataframe))
    dataframe[column_title] = dataframe[column_title].fillna('').astype(str)
    data =  dataframe[dataframe[column_title].str.contains('[A-Za-z]')]
    print('remaining projects with non empty field', column_title, data.shape)  
    
    subset =  data[['PIMS_ID', column_title]]

    
    #reset indeces 
    data = data.reset_index(drop=True)
    subset = subset.reset_index(drop=True)
    
    #rename text column to text
    subset = subset.rename(columns={column_title: "text"})
    print('_________________________')
    print('average lenght of texts:', dataframe[column_title].apply(len).mean())
    print('Median of texts:', dataframe[column_title].apply(len).median())
    
    '''pickle data'''
    #with open(os.path.abspath(os.path.join('..', 'data/interim'))+'/'+column_title+'.pkl', 'wb') as handle:
        #pickle.dump(subset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        #pickle data also for personal QA project:
    # with open(os.path.join('/Users/jonas/Google Drive/github_repos/ClosedDomainQA/data')+'/'+column_title+'.pkl', 'wb') as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return data, subset


def labeling_encoding(dataframe, categories, save = False, add_text = True):
    
    '''
    Function that encodes the labels:
    '''
    
    # generate binary values using MultiLabelBinarizer
    
    df = pd.DataFrame(dataframe, columns=categories)
    
    # append clean text:
    df3 = pd.DataFrame(dataframe, columns=['PIMS_ID', 'all_text_clean', 'all_text_clean_spacy'])
    
    
    for catt in df.columns:
        d = cl.cat(catt)
        list2 = []
        df2 = pd.DataFrame({d: df[catt]})
        
        for i in df2[d].tolist():
            if isinstance(i, str):
                list1 = i.split("; ")
                list1 = [cl.cat(x) for x in list1]
                list2.append(list1)
            else:
                list2.append(['no tag'])
                
        df2[d] = list2
      
        
        multilabel_binarizer = MultiLabelBinarizer()
        multilabel_binarizer.fit_transform(df2[d])
        
        # transform target variable
        y = multilabel_binarizer.transform(df2[d])

        for idx, cat in enumerate(multilabel_binarizer.classes_):
            df2[cat] = y[:,idx]
        
        if add_text == True:        
            result = pd.concat([df3, df2], axis=1, sort=False)
            if save == True:
                print(d)
       
                result.to_csv(os.path.abspath(os.path.join('..', 'data/processed/encoded_labels'))+'/'+ d + ".csv", index=False)
                                 
    return df3, df2, result


def truncate(dataframe, length, save = False, printing = False):
    pims_id = []
    text = []

    for index, row in dataframe.iterrows():

            if len(row['all_text_clean'])  < length:
                pims_id.append(row['PIMS_ID'])
                text.append(row['all_text_clean'])
                
                if printing == True:
                    print('_________________________')
                    print('PIMS ID:', row['PIMS_ID'])

                    print('')
                    print(row['all_text_clean'])
                else:
                    pass
                
    dataframe = dataframe[~dataframe['PIMS_ID'].isin(pims_id)]
       
    if save == True:
        d = {'PIMS_ID':pims_id,'text':text}
        small = pd.DataFrame(d)
        small.to_excel(os.path.abspath(os.path.join('../..', 'data/temp'))+'/less_than_' + str(length) + '.xlsx', index=False)
    else:
        pass
    
    return dataframe



