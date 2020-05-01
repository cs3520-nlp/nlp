import nltk
import random 

#
# file: sentTopic.py
# objective: train a Naive Bayes Classifier 
# on a predefined corpus--'movie reviews' 
#


from nltk.corpus import movie_reviews as mr 
from nltk.tokenize import word_tokenize as wd_split 
from nltk.stem import WordNetLemmatizer


#extract each sentence in movie reviews
#tag each sentence with either positive or negative
documents = []
for category in mr.categories():
    for id in mr.fileids(category): 
        documents.append((list(mr.words(id)), category))
    
#shuffle positive and negative documents
random.shuffle(documents) 

#determine which words are positive and which are negative
every_word = [] 
acceptable_types = ['JJ', 'JJR', 'JJS', 'RBS', 'RBR', 'RB', 'VB']
lemmatizer = WordNetLemmatizer()

#accept words by taking their roots/synonyms 
#only accept words in acceptable types 
for word in mr.words():
    word = lemmatizer.lemmatize(word)
    word = nltk.tag.pos_tag([word])
    for (wd , type) in word: 
        if type in acceptable_types:
            every_word.append(wd)
                
#label each word with their frequency distribution
#set aside the first 1500 words for training purposes
all_freq_words = nltk.FreqDist(every_word)
freq_words = list(all_freq_words.keys())[:1500]

#see which words in in the freq_words 
#are present in each document
def extract_features(doc): 
    words = set(doc)
    features = {} 
    for wd in freq_words:
        word_present = wd in words 
        features[wd] = word_present
    
    return features 

#use extract_features() to get 
#feature sets for each document 
learning_features = [] 
for (doc, sentiment) in documents: 
    learning_features.append((extract_features(doc), sentiment))

#create training and testing sets    
train = learning_features[:1000] 
test = learning_features[1000:]

#train a NaiveBayesClassifier
model = nltk.NaiveBayesClassifier.train(train) 
#print("Model accuracy percent:",(nltk.classify.accuracy(model, test))*100)
#model.show_most_informative_features(15)

#define a function which can evaluate
#the sentiment of a text by using the 
#predefined classifier
def sent_function(text):
    features = extract_features(text)
    return model.classify(features) 
