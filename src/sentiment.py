from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

sentences = [
    "I like pizza.",                    # positive
    "Put the pizza on the table.",      # neutral
    "This pizza is gross.",             # negative
]

analyzer = SentimentIntensityAnalyzer()
for sentence in sentences:
    print(analyzer.polarity_scores(sentence))