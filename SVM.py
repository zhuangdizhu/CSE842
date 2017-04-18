'''
Reference: https://github.com/dkakkar/Twitter-Sentiment-Classifier
'''

import sys
import csv
import os
import re
import nltk
import scipy
import time
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split

def stem(tweet):
        stemmer = nltk.stem.PorterStemmer()
        tweet_stem = ''
        words =[]
        for word in tweet.split():
            if len(word) >= 3:
                if(word[0:2]=='__'):
                    words.append(word)
                else:
                    words.append(word.lower())
        #print(words)
        words2 = []
        for w in words:
            try:
                words2.append(stemmer.stem(w))
            except:
                pass
        tweet_stem = ' '.join(words2)
        return tweet_stem

def preprocessing(tweet):

    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    #Convert @username to __HANDLE
    tweet = re.sub('@[^\s]+','',tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    # Repeating words like happyyyyyyyy
    rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE)
    tweet = rpt_regex.sub(r"\1\1", tweet)
    #Emoticons
    emoticons = \
    [
     ('__positive__',[ ':-)', ':)', '(:', '(-:', \
                       ':-D', ':D', 'X-D', 'XD', 'xD', \
                       '<3', ':\*', ';-)', ';)', ';-D', ';D', '(;', '(-;', ] ),\
     ('__negative__', [':-(', ':(', '(:', '(-:', ':,(',\
                       ':\'(', ':"(', ':((', ] ),\
    ]

    def replace_parenth(arr):
       return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]

    def regex_join(arr):
        return '(' + '|'.join( arr ) + ')'

    emoticons_regex = [ (repl, re.compile(regex_join(replace_parenth(regx))) ) \
            for (repl, regx) in emoticons ]
    for (repl, regx) in emoticons_regex :
        tweet = re.sub(regx, ' '+repl+' ', tweet)
     #Convert to lower case
    tweet = tweet.lower()
    return tweet

def processTweets(X_train, X_test):
        X_train = [stem(preprocessing(line)) for line in X_train]
        X_test = [stem(preprocessing(line)) for line in X_test]
        return X_train, X_test

def classifier(X_train,y_train):
        vec = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf = True,use_idf = True,ngram_range=(1, 2))
        svm_clf =svm.LinearSVC(C=0.1)
        vec_clf = Pipeline([('vectorizer', vec), ('pac', svm_clf)])
        vec_clf.fit(X_train,y_train)
        joblib.dump(vec_clf, 'svmClassifier.pkl', compress=3)
        return vec_clf

def predict(tweet,classifier):

    tweet_processed = stem(preprocessing(tweet))

    if ( ('__positive__') in (tweet_processed)):
         sentiment  = 1
         return sentiment

    elif ( ('__negative__') in (tweet_processed)):
         sentiment  = 0
         return sentiment
    else:

        X =  [tweet_processed]
        sentiment = classifier.predict(X)
        return (sentiment[0])

'''Main'''
reload(sys)
sys.setdefaultencoding("utf-8")

start = time.time()
X = []
y = []
with open("Sentiment Analysis Dataset.csv") as f:
    reader = csv.DictReader(f, delimiter=',')
    #length = len(reader)
    count = 0
    for line in reader:
        #if count <= 100000:
            #if count != 8997:
        X.append(line['SentimentText'])
        y.append(line['Sentiment'])
        #else:
            #y.append(line['Sentiment'])
            #continue
        count += 1

X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_test = processTweets(X_train, X_test)
vec_clf = classifier(X_train, y_train)
y_pred = vec_clf.predict(X_test)

print("Analysis: ")
print("Training: " + str(len(X_train)))
print("Testing: " + str(len(X_test)))
print(sklearn.metrics.classification_report(y_test, y_pred))

end = time.time()
print("Time: " + str(end - start))

'''
m = 0
right = 0
while (m < 10000):
    if y_pred[m] == y_test[m]:
        right += 1
    m += 1

print(right)
'''
