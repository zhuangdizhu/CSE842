#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Author  :   Zhuang Di ZHU
#   E-mail  :   zhuangdizhu@yahoo.com
#   Date    :   17/04/13 15:26:34
#   Desc    :
#
import preprocessor as p
from preprocess_twitter import tokenize
p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED, p.OPT.SMILEY)


def str2label(label):
    if "positive" in label:
        return 1
    elif "negative" in label:
        return -1
    else:
        return 0

def iprint(s1,s2):
    if s2 != None:
        for i in s2:
            print("start-index:",i.start_index)
            print("content:", i.match)

LINE_NUM = 10
input_raw_file = "../dataset2/data/b.traing.tsv"
clean_tweet_file = "../output_set/dataset2.b.test.clean_tweet.txt"
smile_faces = [":_)",":)",": )"]

with open(input_raw_file, 'r') as fp_read:
    with open(clean_tweet_file, 'w') as fp_write:
        for i in range(LINE_NUM):
            line = fp_read.readline().split()
            label = str2label(line[2])
            tweet = " ".join(line[3:])

            #clean_tweet = p.clean(tweet)
            clean_tweet = tokenize(tweet)
            #parsed_tweet = p.parse(tweet)

            # find :-) :) : ) and change them to Happy.
            #for face in smile_faces:
            #    clean_tweet = clean_tweet.replace(face,"I am Happy .")
            ## remove hashtag signs
            #    clean_tweet = clean_tweet.replace("#", "")
            ## remove metion signs
            #    clean_tweet = clean_tweet.replace("@","")
            ## 3> to love
            #    clean_tweet = clean_tweet.replace("3>","love")


            print(clean_tweet)
            #iprint('URL:',parsed_tweet.urls)
            #iprint('Mention:',parsed_tweet.mentions)
            #iprint('HASHTAG:',parsed_tweet.hashtags)
            #iprint('RW:',parsed_tweet.reserved_words)
            #iprint('EMOJI:',parsed_tweet.emojis)
            #iprint('SMILE:',parsed_tweet.smileys)

            output_line = str(label)+"\t"+clean_tweet+"\n"
            try:
                fp_write.write(output_line)
            except:
                print(str(i) +": Discard one message ...")

            #print(tweet+"   " + str(label))
            #print(clean_tweet)

