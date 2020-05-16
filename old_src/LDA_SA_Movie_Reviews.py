#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

data = pd.read_csv('IMDB-Dataset.csv', error_bad_lines=False);

# split positive and negative sentiment reviews
pos_reviews = data[data.sentiment == "positive"]
neg_reviews = data[data.sentiment == "negative"]

pos_data = pos_reviews[['review']]
pos_data['index'] = pos_data.index
pos_documents = pos_data

neg_data = neg_reviews[['review']]
neg_data['index'] = neg_data.index
neg_documents = neg_data

# showing that the reviews were correctly split by sentiment
print(len(pos_documents))
print(pos_documents[:5])
print(len(neg_documents))
print(neg_documents[:5])


# In[2]:


from nltk.corpus import stopwords
import re

stop_words = list(set(stopwords.words('english')))

'''Positive Data'''
# Remove punctuation using regular expresssion
pos_documents['review_processed'] = pos_documents['review'].map(lambda x: re.sub('[,\.!?]', '', x))
# Lowercase the words using regular expresssion
pos_documents['review_processed'] = pos_documents['review'].map(lambda x: x.lower())
'''Negative Data'''
# Remove punctuation using regular expresssion
neg_documents['review_processed'] = neg_documents['review'].map(lambda x: re.sub('[,\.!?]', '', x))
# Lowercase the words using regular expresssion
neg_documents['review_processed'] = neg_documents['review'].map(lambda x: x.lower())


# In[3]:


from wordcloud import WordCloud

print("Word Cloud generated from positive reviews with minimal preprocessing:")
long_string = " ".join(pos_documents.review_processed)

wordcloud = WordCloud().generate(long_string)
image = wordcloud.to_image()
image.show()

print("Word Cloud generated from negative reviews with minimal preprocessing:")
long_string = " ".join(neg_documents.review_processed)
wordcloud = WordCloud().generate(long_string)
image = wordcloud.to_image()
image.show()


# In[4]:


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk

stemmer = SnowballStemmer("english")
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# In[5]:


pos_processed_documents = pos_documents['review_processed'].map(preprocess)
print("Positive reviews after lemmatizing and stemming:")
print(len(pos_processed_documents))
print(pos_processed_documents[:10])

neg_processed_documents = neg_documents['review_processed'].map(preprocess)
print("\nNegative reviews after lemmatizing and stemming:")
print(len(neg_processed_documents))
print(neg_processed_documents[:10])


# In[9]:


# Making Positive and Negative Dictionaries
pos_dictionary = gensim.corpora.Dictionary(pos_processed_documents)
count = 0
print("\nSome random positive words in our dictionary: ")
for k, v in pos_dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break    
neg_dictionary = gensim.corpora.Dictionary(neg_processed_documents)
count = 0
print("\nSome random negative words in our dictionary: ")
for k, v in neg_dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break


# In[10]:


# Making Postiive and Negative LDA Models
pos_dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
pos_bow_corpus = [pos_dictionary.doc2bow(doc) for doc in pos_processed_documents]
pos_lda_model = gensim.models.LdaMulticore(pos_bow_corpus, num_topics=150, id2word=pos_dictionary, passes=10, workers=2)

neg_dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
neg_bow_corpus = [neg_dictionary.doc2bow(doc) for doc in neg_processed_documents]
neg_lda_model = gensim.models.LdaMulticore(neg_bow_corpus, num_topics=150, id2word=neg_dictionary, passes=10, workers=2)


# In[11]:


# Viewing the LDA Model Topic Results
print("\nFirst Ten Positive Review Topics:")
for idx, topic in pos_lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
    if idx>=9:
        break
print("\nFirst Ten Negative Review Topics")
for idx, topic in neg_lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
    if idx>=9:
        break


# In[12]:


from gensim.models import CoherenceModel

# Compute Perplexity
print('\nPositive Perplexity: ', pos_lda_model.log_perplexity(pos_bow_corpus))  # a measure of how good the model is. lower the better.
# Compute Coherence Score
pos_coherence_model_lda = CoherenceModel(model=pos_lda_model, texts=pos_processed_documents, dictionary=pos_dictionary, coherence='c_v')
pos_coherence_lda = pos_coherence_model_lda.get_coherence()
print('Positive Coherence Score: ', pos_coherence_lda)
'''Negative Topics'''
# Compute Perplexity
print('\nNegative Perplexity: ', neg_lda_model.log_perplexity(neg_bow_corpus))  # a measure of how good the model is. lower the better.
# Compute Coherence Score
neg_coherence_model_lda = CoherenceModel(model=neg_lda_model, texts=neg_processed_documents, dictionary=neg_dictionary, coherence='c_v')
neg_coherence_lda = neg_coherence_model_lda.get_coherence()
print('Negative Coherence Score: ', neg_coherence_lda)


# In[ ]:


#import gensim

#'''Positive'''
# Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
#mallet_path = r'C:/MALLET/bin/mallet.bat' # update this path
#pos_ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=pos_bow_corpus, num_topics=30, id2word=pos_dictionary)

# Show Topics
#print(pos_ldamallet.show_topics(formatted=False))
# Compute Coherence Score
#pos_coherence_model_ldamallet = CoherenceModel(model=pos_ldamallet, texts=pos_processed, dictionary=pos_dictionary, coherence='c_v')
#pos_coherence_ldamallet = pos_coherence_model_ldamallet.get_coherence()
#print('\nCoherence Score: ', pos_coherence_ldamallet)

#'''Negative'''
#neg_ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=neg_bow_corpus, num_topics=30, id2word=neg_dictionary)

# Show Topics
#print(neg_ldamallet.show_topics(formatted=False))
# Compute Coherence Score
#neg_coherence_model_ldamallet = CoherenceModel(model=neg_ldamallet, texts=neg_processed, dictionary=neg_dictionary, coherence='c_v')
#neg_coherence_ldamallet = neg_coherence_model_ldamallet.get_coherence()
#print('\nCoherence Score: ', neg_coherence_ldamallet)


# In[13]:


import pyLDAvis
import pyLDAvis.gensim

# Visualize positive topic words
pyLDAvis.enable_notebook()
pos_vis = pyLDAvis.gensim.prepare(pos_lda_model, pos_bow_corpus, pos_dictionary)
pos_vis


# In[14]:


# Visualize negative topic words
pyLDAvis.enable_notebook()
neg_vis = pyLDAvis.gensim.prepare(neg_lda_model, neg_bow_corpus, neg_dictionary)
neg_vis


# In[40]:


def get_sentiment(text):
    count = 0
    pos_score = 0
    neg_score = 0
    #get what positive topics might be related
    bow_vector = pos_dictionary.doc2bow(preprocess(text))
    
    for idx, score in sorted(pos_lda_model[bow_vector], key=lambda tup:-1*tup[1]):
        count+=1
        pos_score += score
        if count > 2:
            break
    pos_score = pos_score/3
    pos_score *= 100
    
    count = 0
    #get what negative topics might be related
    bow_vector = neg_dictionary.doc2bow(preprocess(text))
    for idx, score in sorted(neg_lda_model[bow_vector], key=lambda tup:-1*tup[1]):    
        neg_score += score
        count+=1
        if count > 2:
            break
    neg_score = neg_score/3
    neg_score *= 100
    
    result = 0
    if pos_score>neg_score:
        result = (pos_score - neg_score)/(pos_score + neg_score)
        result *= 100
        result = 50 + (result*2)
        return result
    else:
        result = (neg_score - pos_score)/(pos_score + neg_score)
        result *= 100
        result = 50+(result*2)
        return result


# In[49]:


unseen_movie_description = input("Please enter a movie description to analyze: ")
print("\nWe predict that {}% will like this movie".format(get_sentiment(unseen_movie_description)))

