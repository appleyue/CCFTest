import multiprocessing
import numpy as np
import os
from random import shuffle
from random import sample
from code.util.data_handler import get_corpus, get_logger
import gensim
from sklearn.neural_network import MLPClassifier
from gensim.models import Doc2Vec


logger = get_logger("main.doc2vecMLP")

class TrainDoc2vec(object):
    def __init__(self, size=100, train_file="", test_file="", save_name=""):
        self.size = size
        self.train_file = train_file
        self.test_file = test_file
        self.save_name = save_name

    def error_rate_for_model(self, model, train_docs, test_docs,
                             infer=False, infer_steps=5,
                             infer_subsample=0.1, infer_alpha=0.1):
        error_rates = []
        for l in range(3):
            train_x, train_y = zip(*[(doc.sentiment[l], model.docvecs[doc.tags[0]]) for doc in train_docs])
            mlp = MLPClassifier(hidden_layer_sizes=[self.size, 100],
                                max_iter=1000, alpha=0.6).fit(train_x, train_y)
            test_data = test_docs
            if infer:
                if infer_subsample < 1.0:
                    test_data = sample(test_data, int(infer_subsample * len(test_data)))
                test_x = [model.infer_vector(doc.words, steps=infer_steps,
                                             alpha=infer_alpha) for doc in test_data]
            else:
                test_x = [model.docvecs[doc.tags[0]] for doc in test_data]
            test_predictions = mlp.predict(test_x)
            test_y = [doc.sentiment[l] for doc in test_data]
            corretcs = sum(np.rint(test_predictions == test_y))
            errors = len(test_predictions) - corretcs
            error_rate = float(errors) / len(test_predictions)
            error_rates.append(error_rate)
        return error_rates

    def train(self, infer=False):
        corpus, unlabeled_corpus = get_corpus(train_file=self.train_file,
                                              test_file=self.test_file)
        all_docs = corpus + unlabeled_corpus

        # define doc2vec model
        cores = multiprocessing.cpu_count()
        assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be slow"
        model = Doc2Vec(dm=0, iter=5, size=self.size, negative=5,
                        hs=0, min_count=2, workers=cores)
        model.build_vocab(all_docs)
        logger.info("doc2vec model {}".format(model))
        # get train data
        train_docs = [doc for doc in corpus if doc.split == 'train']
        test_docs = [doc for doc in corpus if doc.split == 'test']

        # train
        alpha, min_alpha, passes = (0.02, 0.01, 5)
        alpha_delta = (alpha - min_alpha) / (passes-1)
        logger.info("Start train doc2vec")
        for epoch in range(passes):
            # shuffle in train
            shuffle(all_docs)
            for doc in all_docs:
                shuffle(doc.words)
            model.alpha, model.min_alpha = alpha, alpha
            model.train(all_docs, total_examples=model.corpus_count, epochs=model.iter)

            # test model
            errs = self.error_rate_for_model(model, train_docs=train_docs, test_docs=test_docs, infer=infer)
            logger.info("[{:.3f} {:.3f} {:.3f}] : {} passes".format(errs[0],
                                    errs[1], errs[2], epoch+1))
            logger.info("completed pass {} at alpha {}".format(epoch+1, alpha))
            alpha -= alpha_delta
        model.save(self.save_name)

if __name__ == "__main__":
    size = [100, 200]
    data_root = '../../data'
    train_file = os.path.join(data_root, 'user_tag_query_train_10gram.txt')
    test_file = os.path.join(data_root, 'user_tag_query_test_10gram.txt')
    for s in size:
        logger.info("="*100)
        logger.info("doc vector size: {}".format(s))
        save_name = os.path.join(data_root, 'doc2vec_%d'%s)
        doc2vec = TrainDoc2vec(size=s, train_file=train_file,
                               test_file=test_file, save_name=save_name)
        doc2vec.train(infer=False)

