#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Author  :   Zhuang Di ZHU
#   E-mail  :   zhuangdizhu@yahoo.com
#   Date    :   17/04/13 14:54:57
#   Desc    :
#

from pycorenlp import StanfordCoreNLP

def label2str(label):
    if label == 1:
        return "positive"
    elif label == -1:
        return "negative"
    else:
        return "neutral"

LINE_NUM = 0
ACC = 0
clean_tweet_file = "../output_set/dataset2.b.test.clean_tweet.txt"

nlp = StanfordCoreNLP('http://localhost:9000')

tweets = list()
true_labels = list()

with open(clean_tweet_file, 'r') as fp_read:
    for line in fp_read:
        LINE_NUM += 1
        line = line.split()
        label = int(line[0].strip())
        tweet = " ".join(line[1:])
        tweets.append(tweet)
        true_labels.append(label)

for i, tweet in enumerate(tweets):
    res = nlp.annotate(tweet,
                       properties={
                           'annotators': 'sentiment',
                           'outputFormat': 'json'
                       })
    s = res["sentences"][0]
    true_label = label2str(true_labels[i])
    predict_label = s["sentiment"]
    if true_label.lower() in predict_label.lower():
        ACC += 1
    print "%d: Predict:[%s]-[%s] Truth:[%s]: %s" % (
        i+1,
        predict_label,
        s["sentimentValue"],
        true_label,
        " ".join([t["word"] for t in s["tokens"]]))

ACC = float(ACC)/LINE_NUM
print("ACCURACY = " + str(ACC))
