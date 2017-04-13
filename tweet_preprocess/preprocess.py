#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Author  :   Zhuang Di ZHU
#   E-mail  :   zhuangdizhu@yahoo.com
#   Date    :   17/04/13 15:26:34
#   Desc    :
#
import preprocessor as p
p.set_options(p.OPT.URL, p.OPT.EMOJI)


def str2label(label):
    if "positive" in label:
        return 1
    elif "negative" in label:
        return -1
    else:
        return 0

LINE_NUM = 10
input_raw_file = "../dataset2/data/b.traing.tsv"
clean_tweet_file = "../output_set/dataset2.b.test.clean_tweet.txt"

with open(input_raw_file, 'r') as fp_read:
    with open(clean_tweet_file, 'w') as fp_write:
        for i in range(LINE_NUM):
            line = fp_read.readline().split()
            label = str2label(line[2])
            tweet = " ".join(line[3:])
            clean_tweet = p.clean(tweet)
            output_line = str(label)+"\t"+clean_tweet+"\n"
            fp_write.write(output_line)
            #print(tweet+"   " + str(label))
            #print(clean_tweet)

