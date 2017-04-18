import numpy as np
import csv
import re
import pandas as pd
from collections import defaultdict
from nltk.tokenize.casual import TweetTokenizer
import csv

def write_summary(data, headers, name):
    out_file = open(name, 'w+')
    writer = csv.DictWriter(out_file, fieldnames=headers, lineterminator = '\n')
    header_row = {}
    for header in headers:
        header_row[header] = header
    writer.writerow(header_row)
    for row in data:
        writer.writerow(row)

def main():
    print 'hLLO'
    vocab = Vocab()
    print parse_data_set(vocab, "Test.csv", steps=10)

class Vocab(object):
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.unknown = '<unk>'
        self.add_word(self.unknown, count=0)

    def add_word(self, word, count=1):
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        self.word_freq[word] += count

    def encode(self, word):
        if word in self.word_to_index:
            return self.word_to_index[word]
        return self.word_to_index['<unk>']

    def decode(self, index):
        return self.index_to_word[index]

def is_http_url(s):
    """
    Returns true if s is valid http url, else false
    Arguments:
    - `s`:
    """
    if re.match('https?://(?:www)?(?:[\w-]{2,255}(?:\.\w{2,6}){1,2})(?:/[\w&%?#-]{1,300})?',s):
        return True
    else:
        return False

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def canon_word(s):
    if is_number(s):
        return '<number>'
    elif is_http_url(s):
        return '<url>'
    elif re.match("^([:=8;]['`\-]*[)P])*$",s) or re.match("^([(/])[:=8;]*$",s):
        return '<smile>'
    elif re.match("^([:=8;]['`\-]*[(/|]oO)*$",s) or re.match("^([)/])[:=8;]*$",s):
        return '<sadface>'
    elif s[0] == '#':
        return '<hashtag>'
    elif s[0] == '@':
        return '<user>'
    else:
        return s

def parse_data_iterator(vocab, filename, delimiter=",", steps=10):
    vocab.add_word('</s>')
    file = open(filename, 'r')
    reader = csv.reader(file, delimiter=delimiter, )
    next(reader)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=False)
    for row in reader:
        curr = []
        encoded = []
        label = [row[1]]
        if(row[1] == 0):
            label.append(1)
        else:
            label.append(0)
        words = tokenizer.tokenize(" ".join(row[3:]))
        for i in range(steps):
            if i < len(words):
                try:
                    words[i] = str(words[i])
                except:
                    words[i] = words[i]
                words[i] = canon_word(words[i])
                vocab.add_word(words[i])
                curr.append(words[i])
            else:
                curr.append('</s>')
        for word in curr:
            encoded.append(vocab.encode(word))
        yield label, curr



def parse_data_set(vocab, filename, delimiter=",", steps=10, max_data=10000):
    # filler character for sentences below length of steps
    vocab.add_word('</s>')
    file = open(filename, 'r')
    reader = csv.reader(file, delimiter=delimiter)
    print("Loading Data Success ...")
    headers = next(reader)
    labels = []
    list_of_train = []
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=False)
    count = 0
    for row in reader:
        count += 1
        if count > max_data:
            break
        curr = []
        label = [row[1]]
        if(int(row[1]) == 0):
            label.append(1)
        else:
            label.append(0)
        labels.append(label)
        words = tokenizer.tokenize(" ".join(row[3:]))
        for i in range(steps):
            if i < len(words):
                try:
                    words[i] = str(words[i])
                except:
                    words[i] = words[i]
                words[i] = canon_word(words[i])
                vocab.add_word(words[i])
                curr.append(words[i])
            else:
                curr.append('</s>')
        list_of_train.append(curr)
        # for now we are going to leave out the else case because batching is too slow for unever lenght sentences
    encoded = [[vocab.encode(word) for word in sentence] for sentence in list_of_train ]
    encoded = np.array(encoded).astype(np.int32)
    labels = np.array(labels).astype(np.int32)
    results = {'labels': labels, 'encoded':encoded}
    print("Parsing Data Success ...")
    return results

def xavier_initializer(shape, **kwargs):
    epsilon = np.sqrt(6) / np.sqrt(np.sum(np.array(shape)))
    out = np.random.uniform(low=-epsilon, high=epsilon, size=shape)
    return out

def create_embedding_matrix(vocab, wv_filename, dims):
    """ Should have filled vocab when you execute this method!!"""
    word_map = pd.DataFrame.from_dict(vocab.index_to_word, orient="index")
    word_vecs = xavier_initializer([len(vocab.index_to_word), dims])
    word_map['word_vecs'] = word_vecs.tolist()
    file = open(wv_filename, 'r')
    processed = set
    for line in file:
        line = line.split()
        word = line[0]
        if word in vocab.word_to_index:
            word_vecs[vocab.word_to_index[word]] = line[1:]
    word_vecs = np.array(word_vecs).astype(np.float32)
    return word_vecs



if __name__ == "__main__":
    main()
