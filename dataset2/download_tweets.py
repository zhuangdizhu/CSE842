#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Author  :   Zhuang Di ZHU
#   E-mail  :   zhuangdizhu@yahoo.com
#   Date    :   17/03/12 11:40:20
#   Desc    :
#

#!/usr/bin/python

import sys
import urllib
import re
import json

from bs4 import BeautifulSoup

import socket
socket.setdefaulttimeout(10)

PRMOMPT_MESSAGE = "Usage: python download_tweets.py <input_file> <output_file>\n"
cache = {}
if len(sys.argv) < 3:
    print(PRMOMPT_MESSAGE)
else:
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    input_file = open(input_file)
    with open(output_file,'w') as output_file_f:
        for line in input_file:
            fields = line.rstrip('\n').split('\t')
            sid = fields[0]
            uid = fields[1]
            tweet = None
            text = "Not Available"
            if cache.has_key(sid):
                text = cache[sid]
            else:
                try:
                    f = urllib.urlopen("http://twitter.com/%s/status/%s" % (uid, sid))
                    #Thanks to Arturo!
                    html = f.read().replace("</html>", "") + "</html>"
                    soup = BeautifulSoup(html)
                    jstt   = soup.find_all("p", "js-tweet-text")
                    tweets = list(set([x.get_text() for x in jstt]))
                    if(len(tweets)) > 1:
                        continue
                    text = tweets[0]
                    cache[sid] = tweets[0]
                    for j in soup.find_all("input", "json-data", id="init-data"):
                            js = json.loads(j['value'])
                            if(js.has_key("embedData")):
                                    tweet = js["embedData"]["status"]
                                    text  = js["embedData"]["status"]["text"]
                                    cache[sid] = text
                                    break
                except Exception:
                    continue
            if(tweet != None and tweet["id_str"] != sid):
                text = "Not Available"
                cache[sid] = "Not Available"
            text = text.replace('\n', ' ',)
            text = re.sub(r'\s+', ' ', text)
            #print json.dumps(tweet, indent=2)
            output_line = "\t".join(fields + [text]).encode('utf-8')
            print(output_line)
            output_file_f.write(output_line+'\n')
    input_file.close()
