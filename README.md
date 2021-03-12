# Setup

Please use Python <= 3.7 to ensure working pickle protocol.

Clone the repo to your local machine:
```
git clone https://github.com/jonas-nothnagel/NLP-Project-Documents.git
```
To run the whole repo, more dependencies are needed and creating multiple virtual environments for the individual steps is recommended. 

For running only the web application install the attached dependencies in a virtual environment:
```
python3.7 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
```
On the first run, the app will download several transformer models (3-4GB) and will store them on your local system. 
To start the application, navigate to the streamlit folder and simply run:
```
streamlit run app.py
```
## Neural Question Answering:
Ask open Question to the project documents using a Neural QA pipeline powered by sentence transformers to build corpus embeddings and ranking of paragraphs. For  retrieval a pre-trained transformer model for extractive QA is applied. Highlighted in html. 
![Farmers Market Finder Demo](https://github.com/jonas-nothnagel/NLP-Project-Documents/blob/main/img/neural_qa.gif)


# NCE Document Classification
Working Repo for building a set of models to automate the classification of project log-frames to a comprehensive taxonomy.

## Classification Experiments
Classify documents/text to over 300 categories of the newly introduced taxonomy and obtain detailed "black-box-algorithm" explanation of your prediction.

[![Classification](https://github.com/SDG-AI-Lab/NCE_Document_Classification/blob/master/img/classification.JPG)](#features)

Experiments are continuously pushed in notebooks to the repo.
Due to computational reasons, deep learning approaches are trained and evaluated on GPU powered notebooks. See specifically: 
* https://cloud.google.com/ai-platform

And for older experiments:
* https://console.paperspace.com/tesl8wodi/notebook/pr1zmah40

## Structured Search VS. ElasticSearch
Try out contextual structured search (with AND/OR) and compare to Elastic Search results. Here you can choose between fuzzy string matching and neural sentence transformers for contextual embeddings.

[![Whoosh](https://github.com/jonas-nothnagel/NLP-Project-Documents/blob/main/img/whoosh.png)](#features)

## Zero Shot Classification
Try an implementation of hugginface's Zero-Shot classification model:
[![zero_shot](https://github.com/jonas-nothnagel/NLP-Project-Documents/blob/main/img/zero_shot.png)](#features)


## Information Extraction 

This work streams aims to turn the unstructured data from the log frames to structured data and allows to explore and aggregate the textual data.
