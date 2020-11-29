"""
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@   Instituto Superior Tecnico  @
@@      PRI - 1st Delivery     @@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@    Dinis Araújo - 86406     @@
@@    Inês Lacerda - 86436     @@
@@    Maria Duarte - 86474     @@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
import datetime
import re
from sklearn.feature_extraction import text


# Creating topic class to represent its structure
class Topic:
    def __init__(self, title, desc, narr):
        self.title = title
        self.desc = desc
        self.narr = narr

# Function that processes qrels file
def red_qrels_file(Q_RELS_TEST="qrels.train.txt"):
    rels_dict = {}
    with open(Q_RELS_TEST) as file:
        for line in file:
            topic_num, doc_id, relevante_bool = line.split(' ')
            relevante_bool = relevante_bool.replace('\n', '')
            if topic_num in rels_dict:
                if relevante_bool == '1':
                    rels_dict[topic_num].append(doc_id)
            else:
                if relevante_bool == '1':
                    rels_dict[topic_num] = [doc_id]
    return rels_dict

# Function that processes topics file
def read_topics_file():
    topics_dic = {}
    f = open(Q_TOPICS_PATH, "r")
    content = f.read()
    content = content.replace("\n", " ")

    for topic in content.split("</top>")[:-1]:
        topic = re.split(r'<title>|<desc>|<narr>',topic)
        num = topic[0].replace("<top>  <num> Number: ", "").replace(" ", "")
        title = preprocessing(topic[1])
        desc = preprocessing(topic[2].replace("Description: ", ""))
        narr = topic[3].replace("Narrative: ", "")
        narr = preprocessing(narr.replace(" </top>", ""))

        topics_dic[num] = Topic(title, desc, narr)
    return topics_dic


#  Function responsible of preprocessing all the tokens:
#  punctuation, lower casing, stopwords, stemming
def preprocessing(content):
    # punctuation
    content = re.sub(r'\W', ' ', content)
    # lower casing
    tokens = nltk.word_tokenize(content.lower())
    # stop words
    if stop_words_flag == 'True':
        stop_words = text.ENGLISH_STOP_WORDS.union(set(stopwords.words('english')))
    else:
        stop_words = []
    # stemming
    ps = PorterStemmer()
    preprocessed_tokens = []
    for t in tokens:
        if t not in stop_words and not re.search(r'\d', t) and len(t) > 3:
            preprocessed_tokens.append(ps.stem(t))
    return preprocessed_tokens


######################
# Reading collection #
######################

#   Returns 2 XML Lists:
#       train_xmls, test_xmls
def read_xml_files(D_PATH):
    #folder = os.listdir(D_PATH)
    train_xmls = {}
    test_xmls = {}
    #for folder in folders:
    xml_file_names = os.listdir(D_PATH)
    for xml_file_name in xml_file_names:
        #print(xml_file_name)
        if os.path.isfile(os.path.join(D_PATH, xml_file_name)) and xml_file_name.find(
                ".xml") != -1:
            xml_file = ET.parse(D_PATH + xml_file_name)
            year, month, day = [int(x) for x in
                                xml_file.getroot().attrib.get('date').split(
                                    '-')]
            date = datetime.date(year, month, day)
            document = ''
            for tag in ['headline', 'byline', 'dateline']:
                for content in xml_file.getroot().iter(tag):
                    if content.text:
                        document += ' ' + content.text
            for content in xml_file.getroot().iter('text'):
                for paragraph in content:
                    document += ' ' + paragraph.text
            key = xml_file.getroot().attrib.get('itemid')
            if date <= DATE_TRAIN_UNTIL:
                train_xmls[key] = preprocessing(document)
            else:
                test_xmls[key] = preprocessing(document)

    return train_xmls, test_xmls

#########################################################
#                     Main   Code                       #
#########################################################

# Just some input variable to run our experiments with the analyses.py file

stop_words_flag = 'True'
Q_PATH = "topics.txt"
Q_TOPICS_PATH = "topics.txt"
Q_RELS_TEST = "qrels.train.txt"
DATE_TRAIN_UNTIL = datetime.date(1996, 9, 30)
