# -*-coding:utf-8-*-
import os
import gensim
from code.util.data_handler import get_logger, get_corpus
from random import shuffle

data_root = '../../data'
logger = get_logger('main.word2vec')
train_file = os.path.join(data_root, 'user_tag_query_train_10gram.txt')
test_file = os.path.join(data_root, 'user_tag_query_test_10gram.txt')
corpus, unlabeled_corpus = get_corpus(train_file=train_file, test_file=test_file)

class MySentences(object):
    def __iter__(self):
        for doc in corpus+unlabeled_corpus:
            shuffle(doc.words)
            yield doc.words

#sentences = MySentences()
#model = gensim.models.Word2Vec(sentences, min_count=50, size=128, sample=0,
                               #workers=4, sg=1, window=30, iter=10)

save_path = os.path.join(data_root, 'w2v_model')
#model.save(save_path)
model = gensim.models.Word2Vec.load(save_path)
print([(s[0], s[1]) for s in model.most_similar(u'南京')])
