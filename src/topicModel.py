import sentTopic as sentiment
import gensim
from gensim import models
from gensim.models import LdaModel, LdaMulticore
from gensim import corpora
from gensim.utils import simple_preprocess
from pprint import pprint
from smart_open import smart_open
import os
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

#
# file: topicModel.py
# objective: Determine which documents fall under which topics.
# Code uses file input.
#

#define useful variables
text_files_dir = 'text_files'
stop_words = set(stopwords.words('english'))
files = (file for file in os.listdir(text_files_dir))
non_acceptable_types = ['CC', 'EX', 'IN', 'WRB', 'MD',]

#function tokenizes files by sentence
def create_documents():
    documents = []
    for file in files:
        text = open(os.path.join(text_files_dir, file), 'r').read()
        sentences = sent_tokenize(text)

        for s in sentences:
            documents.append(s)
    return documents

#function takes an individual file
#and tokenizes by sentence
def create_documents_file(file):
    documents = []
    text = open(os.path.join(text_files_dir, file), 'r').read()
    sentences = sent_tokenize(text)

    for s in sentences:
        documents.append(s)

    return documents

#function preprocesses documents
#ensures only words not in stop_words and > 3 are accepted
def word_tokens(doc):
    result = []
    for w in simple_preprocess(doc):
        word = nltk.tag.pos_tag([w])
        for (w, type) in word:
            if w not in stop_words and len(w) > 3:
                result.append(w)

    return result

#populate variables that will be used in initial training
documents = create_documents()
processed_documents = (word_tokens(doc) for doc in documents)

#make a dictionary
dictionary = corpora.Dictionary(processed_documents)

#make a corpus/bag of words
bags_of_words = [dictionary.doc2bow(w) for w in processed_documents]

#use this term-frequency, inverted document frequency to devalue
#words that occur in every document
tfidf = models.TfidfModel(bags_of_words)
corpus_tfidf = tfidf[bags_of_words]

#train an lda model using corpus_tfidf
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=3,
                                       id2word=dictionary,
                                       passes=2,
                                       workers=2)

#for idx, topic in lda_model_tfidf.print_topics(-1):
    #print('Topic: {} \nWords: {}'.format(idx, topic))

#Example output
#Retrieves file1.txt which is a negative yelp review about plumbing

#populate variables that contain sentences and preprocessed word tokens
documents = create_documents_file('file1.txt')
processed_documents = (word_tokenize(doc) for doc in documents)

#create a new dictionary and a bag of words vector
mydictionary = corpora.Dictionary(processed_documents)
bow_vec = mydictionary.doc2bow(doc for doc in processed_documents)

#use trained model to determine which of 3 topics, bow_vec falls under
##### NEEDS IMPROVEMENT ####
print("\n")

for index, score in sorted(lda_model_tfidf[bow_vec], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))

print("\n")

#use imported sentTopic.py as module
#associate value +1 with pos and -2 with neg (trying to avoid 0)
#running total of each sentence being either pos or neg will determine overall sentiment
answer = 0
for doc in documents:
    sent = sentiment.sent_function(doc)
    if sent == "pos":
        answer+=1
    if sent == "neg":
        answer+=(-2)

if answer > 0:
    sent = "pos"
elif answer < 0:
    sent = "neg"
else:
    sent = "undetected"

#Print result
print("Evaluated topic is: " + sent)

