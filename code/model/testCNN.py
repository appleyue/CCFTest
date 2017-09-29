import os
import numpy as np
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
from code.util.data_handler import get_corpus, get_logger, generate_dictionary
import tensorflow as tf
import keras.backend as K
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D, Convolution1D, Embedding
from tensorflow.contrib.keras.api.keras.utils import to_categorical
from tensorflow.contrib.keras.api.keras.preprocessing import sequence

data_root = "../../data"
logger = get_logger("main.textCNN")

class CNN(object):
    def __init__(self, w2v_file, word_file, max_len=600, w2v_dim=128, batch_size=512, epochs=50):
        self.model = None
        self.w2v = Word2Vec.load(w2v_file)
        self.dictionary = Dictionary.load(word_file)
        self.max_len = max_len
        self.w2v_dim = w2v_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.vocab_size = len(self.dictionary.token2id)
        logger.info("vocab size: {}".format(self.vocab_size))

    def fit(self, train_docs):
        train_x, age_y, gender_y, education_y = [], [], [], []

        for doc in train_docs:
            x = []
            for word in doc.words:
                if self.dictionary.token2id.__contains__(word):
                    x.append(self.dictionary.token2id[word]+1)
            train_x.append(x)
            age_y.append(doc.sentiment[0])
            gender_y.append(doc.sentiment[1])
            education_y.append(doc.sentiment[2])
        logger.info("train size: {}".format(len(train_x)))
        train_x = sequence.pad_sequences(train_x, maxlen=self.max_len, padding="post", truncating="post")

        one_hot_age_y = to_categorical(age_y)
        one_hot_gender_y = to_categorical(gender_y)
        one_hot_education_y = to_categorical(education_y)

        logger.info("init embedding matrix.")
        not_found = 0
        embedding_init = np.zeros((self.vocab_size+1, self.w2v_dim))
        
        for i in range(1, self.vocab_size+1):
            word = self.dictionary.get(i-1)
            if self.w2v.__contains__(word):
                embedding_init[i] = self.w2v[word]
            else:
                not_found += 1
        logger.info("not found {}".format(not_found))

        # CNN model
        input_layer = Input(shape=[self.max_len, ], dtype=tf.float32, name="input_layer")

        mem_embedding_layer = embedding_layer = Embedding(input_dim=self.vocab_size + 1, output_dim=self.w2v_dim,
                                    embeddings_initializer='truncated_normal', input_length=self.max_len, name="embedding_layer")

        embedding_layer = embedding_layer(input_layer)
        cnn_layer = Convolution1D(filters=200, kernel_size=1, padding="valid", activation="tanh", name="cnn_layer")(embedding_layer)

        avg_pool_layer = GlobalAveragePooling1D(name="avg_pool_layer")(cnn_layer)
        dense1_layer = Dense(units=200, activation="relu", name="dense1_layer")(avg_pool_layer)
        dropout1_layer = Dropout(rate=0.3, name="dropout1_layer")(dense1_layer)

        age_layer = Dense(units=100, activation="relu", name="age_layer")(dropout1_layer)
        age_dropout_layer = Dropout(rate=0.3, name="age_dropout_layer")(age_layer)
        age_output = Dense(units=7, activation="softmax", name="age_output")(age_dropout_layer)

        gender_layer = Dense(units=100, activation="relu", name="gender_layer")(dropout1_layer)
        gender_dropout_layer = Dropout(rate=0.3, name="gender_dropout_layer")(gender_layer)
        gender_output = Dense(units=3, activation="softmax", name="gender_output")(gender_dropout_layer)

        education_layer = Dense(units=100, activation="relu", name="education_layer")(dropout1_layer)
        education_dropout_layer = Dropout(rate=0.3, name="education_dropout_layer")(education_layer)
        education_output = Dense(units=7, activation="softmax", name="education_output")(education_dropout_layer)

        self.model = Model(inputs=input_layer, outputs=[age_output, gender_output, education_output])
        age_weights = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
        gender_weights = {0: 0, 1: 1, 2: 1}
        education_weights = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

        self.model.summary()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy',
                           metrics=['accuracy'])
        mem_embedding_layer.set_weights([embedding_init])

        self.model.fit(train_x, [one_hot_age_y, one_hot_gender_y, one_hot_education_y],
                       class_weight=[age_weights, gender_weights, education_weights],
                       epochs=self.epochs, batch_size=self.batch_size)
        return self

    def predict(self, test_docs):
        self.test_docs = test_docs
        test_x = []
        for doc in test_docs:
            x = []
            for word in doc.words:
                if self.dictionary.token2id.__contains__(word):
                    x.append(self.dictionary.token2id[word] + 1)
            test_x.append(x)
        test_x = sequence.pad_sequences(test_x, maxlen=self.max_len, padding="post", truncating="post")
        self.proba = self.model.predict(test_x)
        return self.proba

    def result(self):
        age_test_y, gender_test_y, education_test_y = [], [], []
        for doc in self.test_docs:
            age_test_y.append(doc.sentiment[0])
            gender_test_y.append(doc.sentiment[1])
            education_test_y.append(doc.sentiment[2])

        self.proba[0] = np.delete(self.proba[0], 0, axis=1)
        self.proba[1] = np.delete(self.proba[1], 0, axis=1)
        self.proba[2] = np.delete(self.proba[2], 0, axis=1)

        def label(probability):
            return np.argmax(probability)+1
        correct = np.array([0, 0, 0], dtype=np.float32); test_num = np.array([0, 0, 0], dtype=np.float32)
        pred = [[], [], []]
        for i in range(3):
            pred[i] = map(label, self.proba[i])
        for index, doc in enumerate(self.test_docs):
            for i in range(3):
                if doc.sentiment[i] != 0:
                    test_num[i] += 1
                if pred[i][index] == doc.sentiment[i]:
                    correct[i] += 1
        logger.info("correct num: {}".format(correct))
        logger.info("test_num: {}".format(test_num))
        return [correct/test_num, np.mean(correct/test_num)]

if __name__ == "__main__":
    train_file = os.path.join(data_root, 'user_tag_query_train_10gram.txt')
    test_file = os.path.join(data_root, 'user_tag_query_test_10gram.txt')
    corpus, _ = get_corpus(train_file=train_file, test_file=test_file)
    train_docs = [doc for doc in corpus if doc.split == 'train']
    test_docs = [doc for doc in corpus if doc.split == 'test']

    word_path = os.path.join(data_root, 'word_dictionary')
    w2v_file = os.path.join(data_root, 'w2v_model')
    #generate_dictionary(train_file=train_file, test_file=test_file, save_path=word_path)
    cls = CNN(w2v_file=w2v_file, word_file=word_path, max_len=600, epochs=50)
    #cls.test()
    cls.fit(train_docs=train_docs)
    cls.predict(test_docs=test_docs)
    result = cls.result()
    logger.info("="*50)
    logger.info(result)
    logger.info("="*50)

