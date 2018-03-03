import os
import pandas


def getTweetTextByLine():
    with open(DATA_PATH_TWEET_TEXT) as f:
        content = f.read().splitlines()

    return content


current_dir_path = os.path.dirname(os.path.realpath(__file__))

DATA_PATH_TWEET_IDS = current_dir_path + '/venv/Data/tweet_by_ID_25_2_2018__08_23_08.txt.ids'
DATA_PATH_TWEET_TEXT = current_dir_path + '/venv/Data/tweet_by_ID_25_2_2018__08_23_08.txt.text'
DATA_PATH_TWEET_LABELS = current_dir_path + '/venv/Data/tweet_by_ID_25_2_2018__08_23_08.txt.labels'

list_of_tweets = getTweetTextByLine()

tweet_ids = data = pandas.read_csv(DATA_PATH_TWEET_IDS, names=['id'])
tweet_text = data = pandas.DataFrame({'tweet_text': list_of_tweets})
tweet_labels = data = pandas.read_csv(DATA_PATH_TWEET_LABELS, names=['label'])

print(tweet_ids.shape)
print(tweet_text.shape)
print(tweet_labels.shape)

twitter_data = pandas.concat([tweet_ids, tweet_text, tweet_labels], axis=1)

print(twitter_data.shape)

print(twitter_data)
