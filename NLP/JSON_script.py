# Library imports
import json, os, bz2
#import numpy as np
import pandas as pd
import unicodecsv as csv
import unicodedata
import contextlib # unicodecsv for non-ascii handling
import csv as csv2
import re
from IPython.display import clear_output   # to clear screen between directory prints

from languagedetector import detect_language, UnknownLanguageException # for detecting lannguage from text
def clear_text(text_raw, dict_twitter_names):
 # find out whether company in tweet
 company = []
 for comp, name in dict_twitter_names.items():
   if comp in re.split(' |; |\'|, |: |\*|\n', text_raw):
     company.append(name)
 if company:
   # remove tweets that are not english
   text = ''.join([l for l in text_raw if unicodedata.category(l)[0] not in ('S', 'M', 'C')])
   try:
     # disables error messages momentarily
     with contextlib.redirect_stderr(None):
       # detects the languages
       langs = detect_language(text)
       # list comprehension
       # this could be adjustable as a hyperparameter - languageconfidence
       langs = [l.code for l in langs if l.confidence > 95.]
       if not langs or 'en' not in langs:
         company = []
   except UnknownLanguageException:
     company = []
 return company

# Top level directory - Use forward slashes only (/) - Do not place / at end
directory = "C:/Users/ge57sah/Desktop/12"
# Output .csv table name - Will be placed in same directory as Jupyter script
outfile = 'tweets_eval_2018_12_all.csv'
##################################################
def process_json(comp_dict,directory,outfile):
 # Initializing variables
 cursor=' >> '
 count_mentions=0
 count_replies=0
 count_tweets=0
 with open(outfile, 'wb') as csvfile:
   writer = csv.writer(csvfile)
   # Write header row
   writer.writerow(['poster','recipient','relationship','tweet date','tweet id','tweet','retweet status', 'company'])
   # Walk through all subdirectories
   for dirs, subdirs, files in os.walk(directory):
     # Screen prints
     clear_output()
     print(cursor,'mentions:',count_mentions)
     print (cursor,'replies:',count_replies)
     print (cursor,'tweets:',count_tweets)
     print (cursor,'* total:',count_mentions+count_replies+count_tweets)
     print ('-'*10)
     print (cursor,'currently searching', dirs)
     for file in files:
       if file.endswith('.bz2'):
         # Extract bz2 archives to memory
         file = bz2.BZ2File(os.path.join(dirs, file), 'rb')
         for line in file:
           # Load each record as json object
           try:
             tweet = json.loads(line)
           except json.decoder.JSONDecodeError as e:
             pass
           # Save standard tweet info
           try:
             poster = tweet['user']['screen_name']
             tweet_date = tweet['created_at']
             tweet_id = tweet['id']
             retweet = 'False'
             if tweet['truncated'] == True:
               tweet_text = tweet['extended_tweet']['full_text']
             elif bytearray('retweeted_status', 'utf-8') in line.lower():
               retweet = 'True'
               if tweet['retweeted_status']['truncated'] == True:
                 tweet_text1 = tweet['text'].split(": ", 1)[0]
                 tweet_text2 = tweet['retweeted_status']['extended_tweet']['full_text']
                 tweet_text = tweet_text1 + ": " + tweet_text2
               else:
                 tweet_text = tweet['text']
             elif bytearray('quoted_status', 'utf-8') in line.lower():
               if tweet['quoted_status']['truncated'] == True:
                 tweet_text1 = tweet['text']
                 tweet_text2 = tweet['quoted_status']['extended_tweet']['full_text']
                 tweet_text = tweet_text1 + " QUOTED: " + tweet_text2
               else:
                 tweet_text1 = tweet['text']
                 tweet_text2 = tweet['quoted_status']['text']
                 tweet_text = tweet_text1 + " QUOTED: " + tweet_text2
             else:
               tweet_text = tweet['text']
           except:
             pass
           companies = clear_text(tweet_text, comp_dict)
           if companies:
             try:
               for company in companies:
                 writer.writerow(
                   [poster, poster, 'tweet', tweet_date, tweet_id, tweet_text, retweet,
                    company])
                 count_tweets += 1
             except:
               print('couldnt write')
               pass
         else:
           pass
if __name__ == '__main__':
 comp_dict = pd.read_csv("C:/Users/ge57sah/PycharmProjects/pythonProject/sp500_twitter_names.csv", index_col=0, squeeze=True).to_dict()
 process_json(comp_dict,directory,outfile)
 print ('-'*10)
 print ('complete!')