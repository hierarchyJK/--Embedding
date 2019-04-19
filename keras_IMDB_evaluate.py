# -*- coding:utf-8 -*-
"""
@project: PyCharmProject
@author: KunJ
@file: keras_IMDB_evaluate.py
@ide: Pycharm
@time: 2019-04-19 15:41:39
@month: 四月
"""
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import load_model
from keras.models import Sequential
def dataprocess():
    """
    处理IMDB原始数据的标签
    :return: 测试集集及其标签
    """
    labels = []
    texts = []
    imdb_dir = 'F:\\IMDB数据集\\aclImdb'
    train_dir = os.path.join(imdb_dir, 'test')

    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname), 'r', encoding='utf-8')
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
    return texts, labels
texts, labels = dataprocess()
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=100)
y_test = np.asarray(labels)

model = load_model('pre_trained_glove_model.h5')
loss, acc = model.evaluate(x_test, y_test)
print(loss, acc)
