from sklearn.model_selection import train_test_split
import pandas as pd
import re
from w3lib.html import replace_entities
import ftfy as ft
from w3lib.html import replace_entities

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


full_issue = pd.read_csv(r'E:\IR\data\NEW_full_dataset_wir_alik_fg_ext 211020_pola check_vt.csv', sep=";", engine='python')
#check for duplicates
issue_rows_before = len(full_issue.index)

full_issue.drop_duplicates(subset=['tweet_content', 'polarity'], keep="first", inplace=True)

issue_rows_after = len(full_issue.index)

if issue_rows_after == issue_rows_before:
    print("no duplicates in dataset")

print(full_issue.head())


train_issue = pd.DataFrame()
test_issue = pd.DataFrame()


full_issue.tweet_content = full_issue.tweet_content.apply(ft.fix_text)
full_issue.tweet_content = full_issue.tweet_content.apply(lambda x: clear_text(x))

#could use stratify=y_issue to get evenly distributed datasets across labels but it doesnt work with only one member in a class so cant be done in this case unless class is dropped

# delete those with more than one label or no labels
i = full_issue[(full_issue[['Environment', 'Social Capital', 'Human Capital', 'Business Model & Innovation', 'Leadership & Governance', 'None']].sum(axis=1) > 1) | (full_issue[['Environment', 'Social Capital', 'Human Capital', 'Business Model & Innovation', 'Leadership & Governance', 'None']].sum(axis=1) == 0)].index
full_issue.drop(i, inplace=True)

full_issue.loc[(full_issue["polarity"] == 1), 'polarity']=1
full_issue.loc[(full_issue["polarity"] == -1), 'polarity']=0
full_issue.dropna(subset=['polarity'], inplace=True)


X_issue = full_issue.tweet_content
y_issue = full_issue.polarity

print(full_issue.groupby(['polarity']).size())
print(X_issue.head())
print(y_issue.head())
X_issue_train, X_issue_test, y_issue_train, y_issue_test = train_test_split(X_issue, y_issue, train_size=0.67, random_state=42, stratify=y_issue)

print(f"Train labels issue:\n{y_issue_train}")
print(f"Test labels issue:\n{y_issue_test}")

train_issue = pd.concat([X_issue_train, y_issue_train], axis=1)
test_issue = pd.concat([X_issue_test, y_issue_test], axis=1)

print(train_issue.iloc[0:10,])
print(test_issue.iloc[0:10,])

train_issue.columns = ['text', 'label']
test_issue.columns = ['text', 'label']

train_issue.to_pickle(r'E:\IR\data\pola\NEW_train_wir_alik_fg_ext 281120_pola.pkl')
train_issue.to_csv(r'E:\IR\data\pola\NEW_train_wir_alik_fg_ext 281120_pola.csv', index=False, sep=';', encoding='utf-8-sig')
test_issue.to_csv(r'E:\IR\data\pola\NEW_test_wir_alik_fg_ext 281120_pola.csv', index=False, sep=';', encoding='utf-8-sig')
test_issue.to_pickle(r'E:\IR\data\pola\NEW_test_wir_alik_fg_ext 281120_pola.pkl')

print(train_issue.dtypes)
print(test_issue.dtypes)
