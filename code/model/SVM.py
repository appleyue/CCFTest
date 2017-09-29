from collections import Counter
from sklearn.svm import LinearSVC
import numpy as np
from scipy.sparse import csr_matrix
from code.util.data_handler import get_logger

logger = get_logger('main.SVM')
def compute_ratios(counters, alpha=1.0):
    '''
    :param counters: 
    :param alpha: 
    :return: log-likelihood ratios for each class 
    '''
    ratios = dict()

    # create vocabulary and index dict
    all_ngrams = set()
    for counter in counters.values():
        all_ngrams.update(counter.keys())
    all_ngrams = list(all_ngrams)
    V = len(all_ngrams)
    dic = dict((t, i) for i, t in enumerate(all_ngrams))
    # logger.info("dic length: {}".format(len(dic)))
    # logger.info("all_grams length:{} {}".format(len(all_ngrams), all_ngrams[:10]))
    # calculate NB feature
    # f[i][j] = occurrences number of feature Vj in sample Xi

    # p = alpha + sum(f[i]) if y[i] == 1
    # q = alpha + sum(f[i]) if y[i] == 0
    # r = log(p/||p||1) - log(q/||q||1)
    sum_counts = np.full(V, 2*alpha)
    for c in counters:
        counter = counters[c]
        for t in all_ngrams:
            sum_counts[dic[t]] += counter[t]
    for c in counters:
        counter = counters[c]
        p_c = np.full(V, alpha)
        for t in all_ngrams:
            p_c[dic[t]] += counter[t]
        q_c = sum_counts - p_c

        p_c = p_c / np.linalg.norm(p_c, ord=1) # max(sum(abs(x), axis=0))
        q_c = q_c / np.linalg.norm(q_c, ord=1)
        p_c = np.log(p_c)
        q_c = np.log(q_c)
        ratios[c] = p_c - q_c
        # logger.info( "class:{} ratio:{}".format(c, ratios[c]))
    return dic, ratios, V

def generate_data(traindocs, dic, V, ratios, l, NB=True):
    # for multi class, train multi bi-classify
    n_samples = len(traindocs)
    classes = ratios.keys()
    Y_real = np.zeros(n_samples, dtype=np.int64)
    X = dict()
    Y = dict()
    data = dict()
    for c in classes:
        Y[c] = np.zeros(n_samples, dtype=np.int64)
        data[c] = []

    indptr = [0]; indices = []
    for i, doc in enumerate(traindocs):
        Y_real[i] = doc.sentiment[l]
        for c in classes:
            Y[c][i] = int(doc.sentiment[l] == c)

        for w in doc.words:
            if w in dic:
                index = dic[w]
                indices.append(index)
                for c in classes:
                    if NB:
                        data[c].append(ratios[c][index])
                    else:
                        data[c].append(1)
        indptr.append(len(indices))
    for c in classes:
        X[c] = csr_matrix((data[c], indices, indptr), shape=[n_samples, V],
                          dtype=np.float32)
    return X, Y, Y_real


class SVMClassifier:
    def __init__(self, alpha=1.0, C=0.001, l=0, NB=False, beta=0.25):
        self.alpha = alpha
        self.C = C
        self.l = l
        self.NB = NB
        self.beta = beta
        self.counters = {}
        self.dic, self.ratios, self.v, self.classes = None, None, None, None
        self.svms = dict()

    def fit(self, train_docs):
        for doc in train_docs:
            label = doc.sentiment[self.l]
            if label not in self.counters:
                self.counters[label] = Counter()
            self.counters[label].update(doc.words)
        self.dic, self.ratios, self.v = compute_ratios(self.counters, alpha=1)
        self.classes = self.ratios.keys()
        X_train, Y_train, _ = generate_data(train_docs, self.dic,
                                            self.v, self.ratios, self.l, NB=self.NB)
        for c in self.classes:
            self.svms[c] = LinearSVC(C=self.C)
            self.svms[c].fit(X_train[c], Y_train[c])
        return self

    def predict(self, test_docs):
        X_test, Y_test, y_true = generate_data(test_docs, self.dic,
                                               self.v, self.ratios, self.l, NB=self.NB)
        X_all = [[1] * self.v]
        preds = dict()

        for c in self.classes:
            if self.NB:
                avg = self.svms[c].decision_function(X_all) / self.v
                preds[c] = self.beta * self.svms[c].decision_function(X_test[c]) + \
                           np.array((1-self.beta) * avg[0] * np.sum(X_test[c], axis=1).reshape(X_test[c].shape[0]))[0]
            else:
                preds[c] = self.svms[c].decision_function(X_test[c])
        self.y_true = y_true
        return preds

    def result(self, proba):
        correct = 0; cal_test_num = 0
        test_num = len(self.y_true)
        preds = np.zeros(test_num)
        # logger.info("classes: {} y_real: {}".format(self.classes, self.y_true.shape))
        for idx in range(0, test_num):
            max_score = float('-inf')
            for c in self.classes:
                if proba[c][idx] > max_score:
                    max_score = proba[c][idx]
                    preds[idx] = c
            if preds[idx] == self.y_true[idx] and self.y_true[idx] != 0:
                correct += 1
            if self.y_true[idx] != 0:
                cal_test_num += 1
        return correct, cal_test_num, float(correct)/float(cal_test_num)

if __name__ == "__main__":
    data_root = '../../data'
    train_file = 'user_tag_query_train_10gram.txt'
    test_file = 'user_tag_query_test_10gram.txt'

    logger.info("load corpus.")
    corpus, unlabeled_corpus = get_corpus(os.path.join(data_root, train_file),
                                           os.path.join(data_root, test_file))
    all_docs = corpus + unlabeled_corpus
    train_docs = [doc for doc in corpus if doc.split == 'train']
    test_docs = [doc for doc in corpus if doc.split == 'test']

    correct = np.zeros(3, dtype=np.float32)
    test_num = np.zeros(3, dtype=np.float32)
    precision = np.zeros(3, dtype=np.float32)
    interval = np.linspace(0, 1, 21)
    for beta in interval:
        logger.info("="*50)
        for l in range(0, 3):
            logger.info("nbsvm train.")
            nbsvm = SVMClassifier(l=l, NB=True, beta=beta).fit(choose_corpus(train_docs, l=l))
            logger.info("nbsvm predict.")
            nbsvm_preds = nbsvm.predict(test_docs)
            correct[l], test_num[l], precision[l] = nbsvm.result(nbsvm_preds)
            logger.info("l {} SVM: {} {} {}".format(l+1, correct[l], test_num[l], precision[l]))
        logger.info("beta:{} final precision: {} {}".format(beta, precision, precision.mean()))
