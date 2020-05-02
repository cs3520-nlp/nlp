# nlp

# Citations


 sentTopic.py
 Some code inspired by:
 Harrison Kinsley (PythonProgramming.net)
 Article name: Creating a module for Sentiment Analysis with NLTK
 Article name: Converting words to Features with NLTK
 Python
 https://pythonprogramming.net/sentiment-analysis-module-nltk-tutorial/
 https://pythonprogramming.net/words-as-features-nltk-tutorial/

 topicModel.py
 Some code inspired by:
 Selva Prabhakaran (Machine Learning Plus)
 Article name: Gensim Tutorial – A Complete Beginners Guide
 Python
 https://www.machinelearningplus.com/nlp/gensim-tutorial/

 topicModel.py
 Some code inspired by:
 Susan Li (Medium)
 Article name: Topic Modeling and Latent Dirichlet Allocation (LDA) in Python
 Python
 https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

 # How to use new files:
 The sentTopic.py basically trains a Naive Bayes classifier against a ready-made corpus called "movie_reviews".
 By doing so, we avoid trying to make our dataset for training and testing. Possible improvements we can make
 is showing PythonProgramming.net. It involves using multiple classifiers and determining by vote, whether
 a test data should be identified as positive or negative. It can also provide a confidence score. The
 topicModel.py uses file input. It takes in a folder containing files and iterates through each, creating
 a dictionary and a bag of words. The bag of words is then used to train a Latent Dirichlet Allocation (LDA)
 model. For the sake of output, we take the first file in the folder, and use the model to determine
 which topic it most likely belongs to. By importing sentTopic.py as sent (or other name), we are able
 to use a trained classifier and evaluate the sentiment by sentence in the filename. By arbitrarily
 associating positive sentiments with +1 and negative sentiments with -2, we are able to develop a numeric
 value for the overall positivity or negativity. If the overall < 0, then we say the file is negative. Else
 if the overall > 0, we say the file is positive. Improvements are needed. Possible improvements include
 searching online for readymade text files that we can upload to our computers or simply picking a few topics (i.e.
 different services like restaurant, dentist, electrician)  and finding reviews online (i.e. through yelp) and
 amassing as much data as possible. Overall improvements include using pickle to serializing and deserializing
 documents, preprocessed documents, and/or models to save time while we are making changes to the code here and there.
 This simply saves time, but we must take out any pickle statements before turning in the project.
