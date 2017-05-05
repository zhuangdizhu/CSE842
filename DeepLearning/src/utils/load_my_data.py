#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Author  :   Zhuang Di ZHU
#   E-mail  :   zhuangdizhu@yahoo.com
#   Date    :   17/04/17 15:05:15
#   Desc    :
#
import numpy as np
import sys
sys.path
try:
    sys.path.append("../")
except:
    pass

import config

def str2label(label):
    if "positive" in label:
        return 1
    elif "negative" in label:
        return -1
    else:
        return 0

if len(sys.argv) > 1:
    TRAIN_COUNT = int(sys.argv[1])
else:
    model = config.Config('-')
    TRAIN_COUNT = model.datasize


TEST_COUNT = 10000

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

train_dataset = []
test_dataset = []

with open(input_train_file, "r") as fp_reader:
    headers = next(fp_reader)
    for i, row in enumerate(fp_reader):
        if i < TEST_COUNT:
            test_dataset.append(row)
        elif i < TRAIN_COUNT + TEST_COUNT:
            train_dataset.append(row)
        else:
            break

#indices = list(np.random.permutation(LINE_NUM))
#train_dataset = [train_dataset[i] for i in indices]

for i, row in enumerate(train_dataset):
    line = row.split(',')
    label = int(line[1])
    try:
        train_writer.write(row)
        training_cnt += 1
        training_pos += max(0, label)
    except:
        print(str(i) +": Discard one Training message ...")

for i, row in enumerate(test_dataset):
    line = row.split(',')
    label = int(line[1])
    try:
        test_writer.write(row)
        testing_cnt += 1
        testing_pos += max(0, label)
    except:
        print(str(i) +": Discard one Testing message ...")


print("Train Data:", str(training_cnt), str(training_pos) + " + " + str(training_cnt - training_pos))
print("Test  Data:", str(testing_cnt), str(testing_pos) + " + " + str(testing_cnt - testing_pos))

