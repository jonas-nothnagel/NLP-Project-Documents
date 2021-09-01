# Training the classifiers
Here you find a all training scripts for training the classification models. There is one folder respectively for each label. The individual folders contain several different experiments ranging from simple probabilistic modelling to neural networks using pre-trained transformer language models. 

The final combined training script can be found in the  **classification_tfidf_only** notebook.  The models from this notebook are used for deployment.

If you wish to further reduce the feature space you may run the **classification_tfidf_lsa** notebook that applies LSA dimension reduction before fitting the model. The results are promising but it lacks interpretability. For final deployment it may be used. 