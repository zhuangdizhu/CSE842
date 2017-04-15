import csv
import string
import sys
import math
import nltk
import re

def preprocessing(line):
    #Convert to lower case
    line = line.lower()
    #Convert www.* or https?://* to URL
    line = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',line)
    #Convert @username to AT_USER
    line = re.sub('@[^\s]+','',line)
    #Remove additional white spaces
    line = re.sub('[\s]+', ' ', line)
    #Replace #word with word
    line = re.sub(r'#([^\s]+)', r'\1', line)
    #trim
    line = line.strip('\'"')
    return line

def get_word_in_sentences(dataSets):
    all_words = []
    for (words, sentiment) in dataSets:
        all_words.extend(words)
    return all_words

def get_word_features(word_list):
    word_list = nltk.FreqDist(word_list)
    word_features = word_list.keys()
    return word_features

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features



'''
Main
'''

'''Train part'''
train_pos = 0
train_neg = 0
train_neu = 0
dataArray = []
with open("b.traing.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter = '\t')
    for line in tsvreader:
        #if "neutral" and "objective" not in line[2]:
        dataArray.append((preprocessing(line[3]), line[2]))
#print(dataArray[10])

dataSets = []
for (sentences, sentiment) in dataArray:
    word_filted = [e.replace('\\', '') for e in sentences.split()]
    dataSets.append((word_filted, sentiment))
    if sentiment == "positive":
        train_pos += 1
    elif sentiment == "negative":
        train_neg += 1
    else:
        train_neu += 1
#print(dataSets[10])

word_features = get_word_features(get_word_in_sentences(dataSets))
training_set = nltk.classify.apply_features(extract_features, dataSets)
#print(training_set)
print("Training...")
classifier = nltk.NaiveBayesClassifier.train(training_set)

'''
testing
'''
right = 0
wrong = 0
pos_guess = 0
neg_guess = 0
neu_guess = 0
positive = 0
negative = 0
neutral = 0
total = 0
print("Testing...")
with open("b.test.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter = '\t')
    for line in tsvreader:
        #if "neutral" and "objective" not in line[2]:
        '''Analysis'''
        total += 1
        test_sentiment = line[2]
        if test_sentiment == "positive":
            positive += 1
        elif test_sentiment == "negative":
            negative += 1
        else:
            neutral += 1
        test_sentence = preprocessing(line[3])
        test_sentence_token = [e.replace('\\', '') for e in test_sentence.split()]
        tmp_result = classifier.classify(extract_features(test_sentence_token))

        '''Analysis'''
        if tmp_result == "positive":
            pos_guess += 1
        elif tmp_result == "negative":
            neg_guess += 1
        else:
            neu_guess += 1

        if tmp_result == line[2]:
            right += 1
        else:
            wrong += 1

print("train sentiment:")
print(train_pos, train_neg, train_neu)
print("Test sentiment:")
print(positive, negative, neutral)
print("Guess sentiment:")
print(pos_guess, neg_guess, neu_guess)
print("Result:")
print(total, right, wrong)
print(float(right)/float(total))
