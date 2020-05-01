import nltk
import random 
import pickle


from nltk.corpus import movie_reviews as mr 
from nltk.tokenize import word_tokenize as wd_split 
from nltk.stem import WordNetLemmatizer



documents = []
for category in mr.categories():
    for id in mr.fileids(category): 
        documents.append((list(mr.words(id)), category))
        
random.shuffle(documents) 

every_word = [] 
acceptable_types = ['JJ', 'JJR', 'JJS', 'RBS', 'RBR', 'RB', 'VB']
lemmatizer = WordNetLemmatizer()

for word in mr.words():
    word = lemmatizer.lemmatize(word)
    word = nltk.tag.pos_tag([word])
    for (wd , type) in word: 
        if type in acceptable_types:
            every_word.append(wd)
                
all_freq_words = nltk.FreqDist(every_word)
freq_words = list(all_freq_words.keys())[:1500]

def extract_features(doc): 
    words = set(doc)
    features = {} 
    for wd in freq_words:
        word_present = wd in words 
        features[wd] = word_present
    
    return features 

learning_features = [] 
for (doc, sentiment) in documents: 
    learning_features.append((extract_features(doc), sentiment))

train = learning_features[:1000] 
test = learning_features[1000:]

model = nltk.NaiveBayesClassifier.train(train) 
print("Model accuracy percent:",(nltk.classify.accuracy(model, test))*100)
#model.show_most_informative_features(15)

def sent_function(text):
    features = extract_features(text)
    return model.classify(features) 