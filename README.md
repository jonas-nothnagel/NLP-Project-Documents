# Setup

For streamlit I suggest to use > Python 3.7.

Clone the repo to your local machine:
```
git clone https://github.com/jonas-nothnagel/NLP-Project-Documents
```

Install the dependencies in a virtual environment:
```
python3.7 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

On the first run, the app will download several transformer models. To start the application, navigate to the streamlitfolder and simply run:
```
streamlit run app.py
```

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

## Neural Question Answering:
Basic Neural QA pipeline that uses whoosh for document similarity and a pre-trained transformer model for extractive QA. Highlighted in html. 

[![QA](https://github.com/jonas-nothnagel/NLP-Project-Documents/blob/main/img/neural_qa.png)](#features)

## Zero Shot Classification
Try an implementation of hugginface's Zero-Shot classification model:
[![zero_shot](https://github.com/jonas-nothnagel/NLP-Project-Documents/blob/main/img/zero_shot.png)](#features)


## Information Extraction 

This work streams aims to turn the unstructured data from the log frames to structured data and allows to explore and aggregate the textual data.
