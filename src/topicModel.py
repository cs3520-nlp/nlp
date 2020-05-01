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

dir = 'text_files/'
stop_words = set(stopwords.words('english'))
files = (file for file in os.listdir(dir))
non_acceptable_types = ['CC', 'EX', 'IN', 'WRB', 'MD',]

def create_documents():
    documents = []
    for file in files: 
        text = open(os.path.join(dir, file), 'r').read()
        sentences = sent_tokenize(text)
        
        for s in sentences:
            documents.append(s)
    return documents
    
def create_documents_file(file):
    documents = []
    text = open(os.path.join(dir, file), 'r').read()
    sentences = sent_tokenize(text)
        
    for s in sentences:
        documents.append(s)
        
    return documents
    
def word_tokens(doc):
    result = [] 
    for w in simple_preprocess(doc): 
        word = nltk.tag.pos_tag([w])
        for (w, type) in word: 
            if w not in stop_words and len(w) > 3:
                result.append(w)
            
    return result

documents = create_documents()    
processed_documents = (word_tokens(doc) for doc in documents) 

dictionary = corpora.Dictionary(processed_documents)

bags_of_words = [dictionary.doc2bow(w) for w in processed_documents]

tfidf = models.TfidfModel(bags_of_words)
corpus_tfidf = tfidf[bags_of_words]

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=3, 
                                       id2word=dictionary, 
                                       passes=2, 
                                       workers=2)
                                       
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic)) 


documents = create_documents_file('file1.txt')
processed_documents = (word_tokenize(doc) for doc in documents) 

mydictionary = corpora.Dictionary(processed_documents)
bow_vec = mydictionary.doc2bow(doc for doc in processed_documents)

print("\n") 

for index, score in sorted(lda_model_tfidf[bow_vec], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))

print("\n")

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
    
print("Evaluated topic is: " + sent) 
            