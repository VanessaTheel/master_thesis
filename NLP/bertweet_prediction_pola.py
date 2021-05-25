from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from w3lib.html import replace_entities
import ftfy as ft
from w3lib.html import replace_entities
import logging
import json
import os.path

CUDA_LAUNCH_BLOCKING=1

link = 'E:/IR/data/pola/bertweet pola model/jumping-sweep-10/training/checkpoint-360-epoch-20'

def clear_text(text):

    clean_text = re.sub('RT @[A-Za-z0-9_]+: ', '', text)
    # Remove @accounts from text
    #clean_text = re.sub('@', '', text)
    #clean_text = re.sub('@[A-Za-z0-9_]+', '', text)

    # Remove URLs from Test
    #clean_text = re.sub('http\S+|www.\s+', 'URL', clean_text)

    # Remove Hashtags from text
    #clean_text = re.sub('#', '', clean_text)

    clean_text = replace_entities(clean_text)

    return clean_text

#import data
tweets = pd.read_csv('E:/IR/data/raw_18_Q4.csv', sep=",", engine='python')#, quotechar='"', error_bad_lines=False)
tweets['tweet'] = tweets['tweet'].astype(str)

#check for duplicates
issue_rows_before = len(tweets.index)
tweets.drop_duplicates(keep="first", inplace=True)
issue_rows_after = len(tweets.index)

if issue_rows_after == issue_rows_before:
    print("no duplicates in issue dataset")
else:
    diff = issue_rows_before-issue_rows_after
    print("duplicates removed: ", diff)

#clear test like in split train test
tweets['tweet_content'] = tweets.tweet.apply(lambda x: clear_text(x))
tweets['tweet_content'] = tweets.tweet_content.apply(ft.fix_text)

#csv to list to get right format for predicting
tweet_list = tweets.tweet_content.to_list()

# read file
with open(os.path.join(link, 'model_args.json'), 'r') as myfile:
    data=myfile.read()

# parse file
obj = json.loads(data)
train_args = obj

train_args['reprocess_input_data'] = True
train_args['max_seq_length'] = 128

model = ClassificationModel('bertweet', link, args=train_args)

class_list = ['Negative', 'Positive']

predictions, raw_outputs = model.predict(tweet_list)


final = pd.DataFrame(columns=['tweet_content','prediction'])
final['tweet_content'] = tweet_list
final['prediction'] = predictions
final.loc[(final['prediction'] == 0), 'class_name']= class_list[0]
final.loc[(final['prediction'] == 1), 'class_name']= class_list[1]


tweets = final.to_csv('E:/IR/data/pola/output/bertweet/eval_comp_Q418_SP500_jumping_10_pola.csv')