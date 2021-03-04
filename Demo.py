import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn import svm
import pandas as pd

file_name = 'Books_small.json'


class Sentiment:
    NEGATIVE = 'NEGATIVE'
    NEUTRAL = 'NEUTRAL'
    POSITIVE = 'POSITIVE'


class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else:
            return Sentiment.POSITIVE


reviews = []
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall']))

x = [a.text for a in reviews]
y = [b.score for b in reviews]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

vec = CountVectorizer()
X_train_vectors = vec.fit_transform(X_train)
X_test_vectors = vec.fit_transform(X_test)

reg = LinearRegression()
reg.fit(X_test_vectors, y_train)
y_pred = reg.predict(X_train_vectors)


