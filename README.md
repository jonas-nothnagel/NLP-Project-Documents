# Nature Climate and Energy Taxonomy Classification & Portfolio Analysis Experiments
This Repository contains the data and code to set up the taxonomy classification excercise. It is well documented and structured in a way that it can be simply deployed by following the instructions below.
Apart of the taxonomy classification excersise, this repository also features several experiments:
* Neural Question Answering
* Neural and Fuzzy Structured Search - as an improvement over ElasticSearch
* Zero-Shot Text Classification for unsupervised categorisation.
All these items are further explained and introduced in more detail below.

Finally, the repository also contains a web application written in Python using the streamlit library. Itallows to test all models and tools and make them acessible for each team member.

Contents
========

 * [Why?](#Why?)
 * [Installation](#Installation-Setup)
 * [Data](#Data)
 * [Taxonomy Classification](#Taxonomy-Classification-Excercise)
 * [Neural and Fuzzy Structured Search](#Neural-Search)
 * [Neural Question Answering](#neural-qa)
 * [Zero Shot Classification](#zero-shot)
 * [Web Application](#streamlit)
 * [Next Steps](#next)
 * [Notes](#notes)

## Why

---
## Installation
Please use Python <= 3.7 to ensure working pickle protocol.

Clone the repo to your local machine:
```
https://github.com/SDG-AI-Lab/NCE_Document_Classification.git
```
To only run the web application install the dependencies in a virtual environment:
```
python3.7 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
```
On the first run, the app will download several transformer models and will store them on your local system. 
To start the application, navigate to the streamlit folder and simply run:
```
streamlit run main.py
```
To run the whole repo, more dependencies are needed and creating multiple virtual environments for the individual steps is recommended. 

---

## Data
---

## Taxonomy Classification
Classify documents/text to over 300 categories of the newly introduced taxonomy and obtain detailed "black-box-algorithm" explanation of your prediction.

---

### Text-Classification-Feedback-Loop
Deploys trained ML Text Classification models and allows for user feedback to iterate and improve performance over time. 

Input any text and choose from up to 153 categories for prediction. Obtain results and manually correct predictions. The original text and feedback is stored in a [Firebase](https://firebase.google.com/?hl=de) DataBase and can be used for further model training and tuning.
![Demo](https://github.com/jonas-nothnagel/Text-Classification-Feedback-Loop/blob/main/img/demo_1.gif)

### Detailed explanation of important features that algorithm uses for decision
![Classification](./img/classification.JPG)

---
## Neural and Fuzzy Structured Search
Try out contextual structured search (with AND/OR) and compare to Elastic Search results. Here you can choose between fuzzy string matching and neural sentence transformers for contextual embeddings that understand the context of your queries.

![Whoosh](./img/whoosh.png)

---

## Neural Question Answering
Question answering (QA) is a computer science discipline within the fields of information retrieval and natural language processing (NLP), which is concerned with building systems that automatically answer questions posed by humans in a natural language.
A question answering implementation, usually a computer program, may construct its answers by querying a structured database of knowledge or information, usually a knowledge base. More commonly, question answering systems can pull answers from an unstructured collection of natural language documents.
[Source](https://en.wikipedia.org/wiki/Question_answering).


Try this application to ask open questions to the UNDP project documents using a Neural QA pipeline powered by sentence transformers to build corpus embeddings and ranking of paragraphs. For retrieval a pre-trained transformer model for extractive QA is applied. The results are highlighted in html.

[![QA](https://github.com/jonas-nothnagel/ClosedDomainQA/blob/master/img/neural_qa.gif)](#features)

### Upcoming and Possible Ideas
* Add new data sources. 
* FAISS Indexing (50% Done)
* Telegram Chatbot Implementation.
* Refined Ranking.
* Refined Extractive QA

---


## Zero Shot Classification
Try an implementation of hugginface's Zero-Shot classification model:
![zero_shot](./img/zero_shot.png)

---

## Web Application
To propery communicate the findings and result to the team, a web application has been programmed and hosted with the native python library streamlit. 

---

## Next Steps
Following steps are recommended:

---

## Notes

Experiments are continuously pushed in notebooks to the repo.
Due to computational reasons, deep learning approaches are trained and evaluated on GPU powered notebooks. See specifically: 
* https://cloud.google.com/ai-platform

And for older experiments:
* https://console.paperspace.com/tesl8wodi/notebook/pr1zmah40

---