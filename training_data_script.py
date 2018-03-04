from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
import os
from sklearn.naive_bayes import MultinomialNB

current_dir_path = os.path.dirname(os.path.realpath(__file__))

TRAINING_TWEET_DATA_PATH = current_dir_path + '/tweet_texts_training.csv'
TRAINING_LABEL_DATA_PATH = current_dir_path + '/tweets_labels_training.csv'

tweets = []
labels = []

with open(TRAINING_TWEET_DATA_PATH) as f:
    tweets = f.read().splitlines()

with open(TRAINING_LABEL_DATA_PATH) as f:
    labels = f.read().splitlines()

print(len(tweets))
print(len(labels))

vectorizer = CountVectorizer(
    lowercase=True,
    ngram_range=(1, 3),
    analyzer='word',
    tokenizer=nltk.word_tokenize)

print('step 1')

features = vectorizer.fit_transform(
    tweets
)

# tfidf_transformer = TfidfTransformer()
# features_nd = tfidf_transformer.fit_transform(features)
print('step 2')

features_nd = features.toarray()

print('step 3')

X_train, X_test, y_train, y_test = train_test_split(
    features_nd,
    labels,
    train_size=0.8,
    test_size=0.2)

print('step 4')
#
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)

print('step 4')

y_pred = log_model.predict(X_test)

# clf = MultinomialNB().fit(X_train, y_train)
#
# y_pred = clf.predict(X_test)


print(accuracy_score(y_test, y_pred))

print('FINISHED')
