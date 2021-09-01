import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from math import*
import csv


#setting visual options for editors
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
df = pd.read_excel("data_table.xlsm") #excel dataset input

landscape_df = pd.DataFrame(df,columns=['PROJECT DESCRIPTION','Landscapes 1', 'Landscapes 2', 'Landscapes 3'])

#total values
vec = CountVectorizer()
X = vec.fit_transform(landscape_df["PROJECT DESCRIPTION"].values.astype('U'))

total_features = len(vec.get_feature_names())
total_rows = len(landscape_df.index)



#lemmatizer
lemmatizer = WordNetLemmatizer()

#cleaning dataset
def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text


def lemmatize(corpus):
    corpus = remove_stop_words(corpus)
    result = []
    for token in gensim.utils.simple_preprocess(corpus):
        if len(token) > 3:
            result.append(lemmatizer.lemmatize(token))
    return result


def get_frequency_list(file):
    frequency_list = pd.read_csv(file+'.csv', header=None, index_col=0, squeeze=True).to_dict()
    return frequency_list

#trains data and creates frequency list of category into a csv file
#so the csv file would be the trained data
def train_output(dataset, column_1, column_2, column_3, category):

    #print('category: ', category)
    category_df1 = (dataset.loc[dataset[column_1].str.contains(category, na = False)])
    category_df2 = (dataset.loc[dataset[column_2].str.contains(category, na = False)])
    category_df3 = (dataset.loc[dataset[column_3].str.contains(category, na = False)])
    category_row_len = len(category_df1.index+len(category_df2.index)+len(category_df3.index))
    category_probability = category_row_len/total_rows

    concatenated = pd.concat([category_df1, category_df2,category_df3])

    vec_forest = CountVectorizer()
    x_forest = vec_forest.fit_transform(lemmatize(concatenated["PROJECT DESCRIPTION"].values.astype('U'))) #check excel column names
    word_list_forest = vec_forest.get_feature_names();
    count_list_forest = x_forest.toarray().sum(axis=0)
    freq_forest = dict(zip(word_list_forest,count_list_forest))

    w = csv.writer(open(category+"_freq.csv", "w"))
    for key, val in freq_forest.items():
        w.writerow([key, val])

    category_features_count = count_list_forest.sum(axis=0)

    with open(category+'_features_count.txt', 'w') as fileW:
        fileW.write('%d' %category_features_count)

    with open(category+'_probability.txt', 'w') as fileW:
        fileW.write('%f' %category_probability)

category_list = ["Forest", "Conserved_Areas", "Tundra", "FreshWater", "Marine", "Grassland", "Wetlands",
                 "Desert", "Human_altered_Areas"]

#opens up the trained data 
def get_category_features_count(file):
    file = open(file, 'r')
    count = file.readline()
    return int(count)

def get_category_probability(file):
    file = open(file, 'r')
    prob = file.readline()
    return float(prob)

#runs based on training data
def run_training(category_list):
    for i in category_list:
        train_output(landscape_df, 'Landscapes 1', 'Landscapes 2', 'Landscapes 3', i)


#ML algorithm/calculation - Naive Bayes
def calculate(frequency_data, category_features_count, category_probability, input_sentence):
    prob_forest_with_ls = []
    new_word_list = lemmatize(input_sentence)

    #laplace smoothing
    for word in new_word_list:
        if word in frequency_data.keys():
            count = frequency_data[word]
        else:
            count = 0
        prob_forest_with_ls.append((count + 1)/(category_features_count + total_features))

    result_1 = dict(zip(new_word_list,prob_forest_with_ls))
    result_2 = (sum(log(result_1[p]) for p in result_1))
    final_probability = result_2 + log(category_probability)
    return final_probability

#top categories of landscape
frequency_forest = get_frequency_list("Forest_freq")
frequency_conserved = get_frequency_list("Conserved_Areas_freq")
frequency_freshWater = get_frequency_list("FreshWater_freq")
frequency_grassland = get_frequency_list("Grassland_freq")
frequency_human_altered = get_frequency_list("Human_altered_Areas_freq")
frequency_marine = get_frequency_list("Marine_freq")
frequency_tundra = get_frequency_list("Tundra_freq")
frequency_wetlands = get_frequency_list("Wetlands_freq")
frequency_desert = get_frequency_list("Desert_freq")

frequency_list = [frequency_forest,frequency_conserved,frequency_tundra,frequency_freshWater,
                  frequency_marine,frequency_grassland,frequency_wetlands,frequency_desert,
                  frequency_human_altered]

machine_result = []

def result_data(dataset, column):
    for index,row in dataset.iterrows():
        probability_list = {}
        text = row["PROJECT DESCRIPTION"]
        for i in range(0,len(category_list)):

            #print("Category: ", category_list[i])
            category = category_list[i]
            probability = calculate(frequency_list[i],get_category_features_count(category+'_features_count.txt'),
                                              get_category_probability(category+'_probability.txt'),text)
            probability_list[category] = probability

        probability_list  = (sorted(probability_list.items(), key=lambda x: x[1], reverse=True))
        machine_result.append([probability_list[0][0], probability_list[1][0],probability_list[2][0]])

    return machine_result


#below are examples of calling the function for output of machine trained data
#print(result_data(landscape_df,'Landscapes 1'))

#landscape_df1 = pd.DataFrame(landscape_df,columns=['PROJECT DESCRIPTION','Landscapes 1'])
#landscape_df2 = pd.DataFrame(landscape_df,columns=['PROJECT DESCRIPTION','Landscapes 2'])
#landscape_df3 = pd.DataFrame(landscape_df,columns=['PROJECT DESCRIPTION','Landscapes 3'])


#machine_result_landscape_1 = result_data(landscape_df1, "Landscapes 1")
#landscape_df1.insert(2, "machine result_landscape_1", machine_result_landscape_1, True)

# machine_result_landscape_2 = result_data(landscape_df2, "Landscapes 2")
# landscape_df2.insert(2, "machine result_landscape_2", machine_result_landscape_2, True)
# landscape_df2.to_excel("machine_result_output_landscape2.xlsx")

# print(landscape_df3)
# machine_result_landscape_3 = result_data(landscape_df3, "Landscapes 3")
# landscape_df3.insert(2, "machine result_landscape_3", machine_result_landscape_3, True)


#landscape_df1.to_excel("machine_result_output_landscape1.xlsx")
#landscape_df2.to_excel("machine_result_output_landscape2.xlsx")
#landscape_df3.to_excel("machine_result_output_landscape3.xlsx")
