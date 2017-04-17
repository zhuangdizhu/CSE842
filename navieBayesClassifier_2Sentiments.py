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
trainingArray = []
testingArray = []
with open("Sentiment Analysis Dataset.csv") as f:
    reader = csv.DictReader(f, delimiter=',')
    #length = len(reader)
    count = 0
    for line in reader:
        if count > 11000:
            continue
        if count < 10000:
            trainingArray.append((preprocessing(line['SentimentText']), line['Sentiment']))
        else:
            testingArray.append((preprocessing(line['SentimentText']), line['Sentiment']))
        count += 1
#print(trainingArray[10])

dataSets = []
for (sentences, sentiment) in trainingArray:
    word_filted = [e.replace('\\', '') for e in sentences.split()]
    dataSets.append((word_filted, sentiment))
    if sentiment == "1":
        train_pos += 1
    else:
        train_neg += 1
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
positive = 0
negative = 0
total = 0
print("Testing...")
for (test_sentence, test_sentiment) in testingArray:
    #if "neutral" and "objective" not in line[2]:
    '''Analysis'''
    total += 1
    if test_sentiment == "1":
        positive += 1
    else:
        negative += 1
    test_sentence_token = [e.replace('\\', '') for e in test_sentence.split()]
    tmp_result = classifier.classify(extract_features(test_sentence_token))

    '''Analysis'''
    if tmp_result == "1":
        pos_guess += 1
    else:
        neg_guess += 1

    if tmp_result == test_sentiment:
        right += 1
    else:
        wrong += 1

        
print("Analysis:")
print("train sentiment:")
print("total: " + str(train_pos + train_neg))
print("pos: " + str(train_pos) + "; neg: " + str(train_neg))
print("Test sentiment:")
print("total: " + str(positive + negative))
print("pos: " + str(positive) + "; neg: " + str(negative))
print("Guess sentiment:")
print("pos: " + str(pos_guess) + "; neg: " + str(neg_guess))
print("Result:")
print("total: " + str(total))
print("right_guess: " + str(right) + "; wrong_guess: " + str(wrong))
print("Accuracy: " + str(float(right)/float(total)))
