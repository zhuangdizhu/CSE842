"""
preprocess-twitter.py

python preprocess-twitter.py "Some random text with #hashtags, @mentions and http://t.co/kdjfkdjf (links). :)"

Script for preprocessing tweets by Romain Paulus
with small modifications by Jeffrey Pennington
with translation to Python by Motoki Wu

Translation of Ruby script to create features for GloVe vectors for Twitter data.
http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""

import sys
import re

FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        #result = "<hashtag> {} <allcaps>".format(hashtag_body)
        result = "{}".format(hashtag_body)
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
        #result = " ".join(re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

def allcaps(text):
    text = text.group()
    #return text.lower() + " <allcaps>"
    return text.lower()


def tokenize(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    #text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "")

    text = re_sub(r"/"," / ")

    text = re_sub(r"@\w+", "<user>")

    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")

    text = re_sub(r"{}{}p+".format(eyes, nose), "<laugh>")

    #text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sad>")

    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neural>")

    text = re_sub(r"<3","<heart>")

    #text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "")

    text = re_sub(r"#\S+", hashtag)

    #text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"([!?.]){2,}", r"\1 ")

    #text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
    text = re_sub(r"\b(\S*?)(.)\2{3,}(\S*?)\b", r"\1\2\2\3 ")

    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2")

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return text.lower()

def test():
    text1 = "Last day at school. Hello holiday ... #Happpppy"
    text2 = "This website is amazing !!! https://Twitter.com #just #kidding"
    text3 = "@MrL you are so cooooooooool :-) <3"
    text_list = [text2, text3]
    for i in range(2):
        text = text_list[i]
        print(text)
        tokens = tokenize(text)
        print tokens


