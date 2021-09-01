# Source Folder
This folder contains the bread and butter of the whole project: The **tools and helper functions**. Note that most notebooks merely load those functions and run them inside. 

## Processing of Data:
A set of functions to read in the data, clean, normalise and process it.
* import and transform data
* clean data
* process data for Machine Learning

## Exploration:
A set of functions to explore and visual the data.
* Cluster approaches 
* Descriptive Statistics
* Word Clouds

## Feature Engineering:
A set of function to vectorise and embed text.

* Traditional: tf-idf, bow, etc
* Contextual: word2vec, doc2vec, glove, fasttext, etc
* Transformer based (transfer-learning): pre-trained transformer models to embed text and fine-tuning scripts

## Model Training and Evaluation:
Scripts that define model architectures and parameters

* Probabilistic Modelling : Sklearn - Logist Regression, XGBoost, SVM, etc
* Deep Learning: Neural Networks, simple transfomers, etc (Note that these scripts have been removed from here and stored on the cloud environments for GPU support). 
