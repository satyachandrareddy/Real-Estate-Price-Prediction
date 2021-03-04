import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import numpy as np

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
        # print(review['reviewText'])
        # print(review['overall'])
        # reviews.append((review['reviewText'], review['overall']))
        reviews.append(Review(review['reviewText'], review['overall']))

# print(reviews[5])
# print(reviews[5].score)
# print(reviews[5].sentiment)

training, test = train_test_split(reviews, test_size=0.33, random_state=42)
# print(training[0].sentiment)

train_x = [x.text for x in training]
train_y = [x.sentiment for x in training]
# print(train_x[0])
test_x = [x.text for x in test]
test_y = [x.sentiment for x in test]

vectorizer = CountVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
print(train_x_vectors.shape[1])
# print(train_x_vectors[0])
# print(train_x_vectors[0].toarray())
test_x_vectors = vectorizer.fit_transform(test_x)
print(test_x_vectors.shape[1])

clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)
clf_svm.predict(test_x_vectors[0])
