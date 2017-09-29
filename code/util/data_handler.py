import codecs
import random
import os
import numpy as np
import logging
from collections import Counter, namedtuple
from gensim.corpora.dictionary import Dictionary
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('main.data_handler')
Corpus = namedtuple("Document", "words tags split sentiment key")

data_root = '../../data'

def get_logger(name):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(name)
    return logger

def transform(list, keys, age, gender, education):
    validate_ratio = 0.2
    for i in range(len(list)):
        if random.random() >= validate_ratio:
            yield Corpus(words=list[i], tags=[keys[i]], key=keys[i],
                        split='train', sentiment=[age[i], gender[i], education[i]])
        else:
            yield Corpus(words=list[i], tags=[keys[i]], key=keys[i],
                         split='test', sentiment=[age[i], gender[i], education[i]])

def transform_unlabeled(list, keys):
    for i in range(len(list)):
        yield Corpus(words=list[i], tags=[keys[i]], key=keys[i],
                     split='unlabeled', sentiment=[0, 0, 0])

def get_corpus(train_file='',
                test_file='',
                print_counter=False,
                save_file=False):
    raw_train_texts = []
    raw_test_texts = []

    with codecs.open(train_file, 'r', 'gb18030', buffering=1) as handle:
        for line in handle:
            l = [word for word in line.strip("\n").split()]
            raw_train_texts.append(l)
    with codecs.open(test_file, 'r', 'gb18030', buffering=1) as handle:
        for line in handle:
            l = [word for word in line.strip("\n").split()]
            raw_test_texts.append(l)

    train_keys = [l[0] for l in raw_train_texts]
    test_keys = [l[0] for l in raw_test_texts]
    train_age = [int(l[1]) for l in raw_train_texts]
    train_gender = [int(l[2]) for l in raw_train_texts]
    train_education = [int(l[3]) for l in raw_train_texts]

    if save_file:
        np.save(os.path.join(data_root, "testKeys"), test_keys)
        np.save(os.path.join(data_root, "trainKeys"), train_keys)
        np.save(os.path.join(data_root, "trainAge"), train_age)
        np.save(os.path.join(data_root, "trainGender"), train_gender)
        np.save(os.path.join(data_root, "trainEducation"), train_education)

    if print_counter:
        logger.info("age:{}".format(Counter(train_age)))
        logger.info("gender:{}".format(Counter(train_gender)))
        logger.info("education:{}".format(Counter(train_education)))

    for words in raw_train_texts:
        for i in range(4):
            words.pop(0)
    for words in raw_test_texts:
        for i in range(1):
            words.pop(0)

    train_texts = raw_train_texts
    test_texts = raw_test_texts

    corpus = list(transform(train_texts, train_keys, train_age, train_gender, train_education))
    test_corpus = list(transform_unlabeled(test_texts, test_keys))
    return corpus, test_corpus

'''
choose train data that label is not 0
'''
def choose_corpus(train_docs, l):
    res = []
    for doc in train_docs:
        if doc.sentiment[l] != 0:
            res.append(doc)
    return res

def generate_dictionary(train_file="",
                        test_file="",
                        save_path=""):
    train_texts = []
    test_texts = []
    with codecs.open(train_file, 'r', 'gb18030', buffering=1) as handle:
        for ln in handle:
            train_texts.append([word for word in ln.strip("\n").split()[4:]])
    with codecs.open(test_file, 'r', 'gb18030', buffering=1) as handle:
        for ln in handle:
            test_texts.append([word for word in ln.strip("\n").split()[1:]])
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in train_texts+test_texts:
        for token in text:
            frequency[token] += 1
    for text in train_texts:
        for token in list(text):
            if frequency[token] < 50:
                text.remove(token)
    dictionary = Dictionary(train_texts)
    dictionary.save(save_path)
    logger.info(dictionary.get(1))

