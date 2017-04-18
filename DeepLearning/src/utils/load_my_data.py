#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Author  :   Zhuang Di ZHU
#   E-mail  :   zhuangdizhu@yahoo.com
#   Date    :   17/04/17 15:05:15
#   Desc    :
#
import numpy as np

def str2label(label):
    if "positive" in label:
        return 1
    elif "negative" in label:
        return -1
    else:
        return 0

LINE_NUM = 50000
input_train_file = "../../../twitter-sentiment-cnn-master/twitter-sentiment-dataset/sentiment-dataset.csv"
output_train_file = "training.csv"
output_test_file = "testing.csv"

random_rate = 0.2

training_cnt = 0
training_pos = 0

testing_cnt = 0
testing_pos = 0

train_writer = open(output_train_file, 'w')
test_writer = open(output_test_file, 'w')

indices = list(np.random.permutation(LINE_NUM))
dataset = []
with open(input_train_file, "r") as fp_reader:
    headers = next(fp_reader)
    for i, row in enumerate(fp_reader):
        if i >= LINE_NUM:
            break
        dataset.append(row)

dataset = [dataset[i] for i in indices]

for i, row in enumerate(dataset):
    line = row.split(',')
    label = int(line[1])
    if i < LINE_NUM * 0.2:
        try:
            test_writer.write(row)
            testing_cnt += 1
            testing_pos += max(0, label)
        except:
            print(str(i) +": Discard one message ...")
    else:
        try:
            train_writer.write(row)
            training_cnt += 1
            training_pos += max(0, label)
        except:
            print(str(i) +": Discard one message ...")
print("Train Data:", str(training_cnt), str(training_pos) + " + " + str(training_cnt - training_pos))
print("Test  Data:", str(testing_cnt), str(testing_pos) + " + " + str(testing_cnt - testing_pos))

