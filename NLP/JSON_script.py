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
       #print("Language Detection Worked")
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
# Modify keywords, top-level directory, and output file name
##################################################
#with open('C:/Users/Flora/PycharmProjects/IR/data/keywords.csv', newline='') as f:
#  reader = csv2.reader(f)
#  tmp = list(reader)
#keywords = []
#for sublist in tmp:
#  for item in sublist:
#    keywords.append(item)
# Keep keywords lower case - Will match all cases
#keywords = ['ghg emissions', 'greenhouse gas', 'emissions']
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
             #global errors
             #errors += 1
             pass
           # Save standard tweet info
           # print(tweet)
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
                 #print("truncated RT")
               else:
                 tweet_text = tweet['text']
             elif bytearray('quoted_status', 'utf-8') in line.lower():
               if tweet['quoted_status']['truncated'] == True:
                 tweet_text1 = tweet['text']
                 tweet_text2 = tweet['quoted_status']['extended_tweet']['full_text']
                 tweet_text = tweet_text1 + " QUOTED: " + tweet_text2
                 #print('truncated QUOTE')
               else:
                 tweet_text1 = tweet['text']
                 tweet_text2 = tweet['quoted_status']['text']
                 tweet_text = tweet_text1 + " QUOTED: " + tweet_text2
             else:
               tweet_text = tweet['text']
             # Test for retweet status
             # if bytearray('retweeted_status', 'utf-8') in line.lower():
             #  retweet='True'
             # else:
           except:
             pass
           companies = clear_text(tweet_text, comp_dict)
           if companies:
             try:
               # Test for reply relationship
               #if not tweet['in_reply_to_screen_name'] is None:
               #  writer.writerow(
               #    [poster, tweet['in_reply_to_screen_name'], 'reply', tweet_date, tweet_id,
               #     tweet_text,
               #     hashes, retweet, kw])
               #  reply_status = 1
                #  count_replies += 1
               # Test for mention relationships
               #mentions = list()
               #for mention in tweet['entities']['user_mentions']:
               #  recipient = mention['screen_name']
                 # Ensure the mention is not already a reply
               #  if recipient != tweet['in_reply_to_screen_name']:
                #    writer.writerow(
                #      [poster, recipient, 'mentions', tweet_date, tweet_id, tweet_text, hashes,
                #      retweet, kw])
                #    reply_status = 1
                #   count_mentions += 1
               # Write relationship as tweet if no reply or mentions
               #if reply_status == 0:
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
 #comp_dict = pd.read_csv("P:/IR_Twitter Data/Eval_comp/sp500_twitter_names_eval.csv", index_col=0, squeeze=True).to_dict()
 #comp = pd.read_csv("P:/IR_Twitter Data/Eval_comp/sp500_twitter_names.csv", skiprows = 1, header = None)
 #comp = comp.set_index(0)
 #comp_dict = comp.to_dict()
 comp_dict = pd.read_csv("C:/Users/ge57sah/PycharmProjects/pythonProject/sp500_twitter_names.csv", index_col=0, squeeze=True).to_dict()
 process_json(comp_dict,directory,outfile)
 print ('-'*10)
 print ('complete!')