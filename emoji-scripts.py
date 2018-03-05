import os
import pandas

current_dir_path = os.path.dirname(os.path.realpath(__file__))

DATA_PATH_TWEET_IDS = current_dir_path + '/venv/Data/tweet_by_ID_25_2_2018__08_23_08.txt.ids'
DATA_PATH_TWEET_TEXT = current_dir_path + '/venv/Data/tweet_by_ID_25_2_2018__08_23_08.txt.text'
DATA_PATH_TWEET_LABELS = current_dir_path + '/venv/Data/tweet_by_ID_25_2_2018__08_23_08.txt.labels'


def getTweetTextByLine():
    with open(DATA_PATH_TWEET_TEXT) as f:
        content = f.read().splitlines()

    return content


def save_training_data_to_file():
    list_of_tweets = getTweetTextByLine()

    tweet_ids = pandas.read_csv(DATA_PATH_TWEET_IDS, names=['id'])
    tweet_text = pandas.DataFrame({'tweet_text': list_of_tweets})
    tweet_labels = pandas.read_csv(DATA_PATH_TWEET_LABELS, names=['label'])

    top_3_emojis_tweet_data = pandas.concat([tweet_text, tweet_labels], axis=1);
    print(top_3_emojis_tweet_data.shape)

    top_3_emojis_tweet_data = top_3_emojis_tweet_data[top_3_emojis_tweet_data['label'] < 3]

    print(top_3_emojis_tweet_data.shape)

    top_3_emojis_tweet_data['tweet_text'].head(12500).to_csv("tweet_texts_training.csv", encoding='utf-8', index=False, header=False)
    top_3_emojis_tweet_data['label'].head(12500).to_csv("tweets_labels_training.csv", encoding='utf-8', index=False, header=False)

    # print(tweet_ids.shape)
    # print(tweet_text.shape)
    # print(tweet_labels.shape)

    twitter_data = pandas.concat([tweet_ids, tweet_text, tweet_labels], axis=1)

    # training_data = pandas.concat([tweet_text, tweet_labels], axis=1)
    #
    # training_data.to_csv("twitter_training_data.csv", encoding='utf-8', index=False)
    # print(training_data)
    # print(training_data.shape)


save_training_data_to_file()
