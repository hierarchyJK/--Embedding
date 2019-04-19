# -*- coding:utf-8 -*-
"""
@project: PyCharmProject
@author: KunJ
@file: keras_IMDB.py
@ide: Pycharm
@time: 2019-04-19 10:40:51
@month: 四月
"""
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

def dataprocess():
    """
    处理IMDB原始数据的标签,将训练评论转换成字符串列表，每个字符串对应一条评论,
    评论标签（正面 / 负面）转换成 labels 列表
    :return:
     texts：训练集列表
     labels：标签列表
    """
    labels = []
    texts = []
    imdb_dir = 'F:\\IMDB数据集\\aclImdb'
    train_dir = os.path.join(imdb_dir, 'train')

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
print(np.array(texts))
print(np.array(labels))
print(len(texts))
print(len(labels))
print("---------")
maxlen = 100  # 在100个单词后截断评论
training_samples = 200  # 在200个样本上训练
validation_samples = 10000  # 在10000个样本上验证
max_words = 10000  # 只考虑数据集中前10000个最常用见的单词

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)#

sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
# Found 88582 unique tokens.
print("Found %s unique tokens." % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
# Shape of data tensor: (25000, 100)
print('Shape of data tensor:', data.shape)
# Shape of label tensor: (25000,)
print('Shape of label tensor:', labels.shape)

"""将数据划分为训练集和验证集，但首先要打乱数据，因为一开始数据中的样本是拍好序的（所有负面评价都在前面，然后是所有正面的评论）"""
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[: training_samples]
y_train = labels[: training_samples]
x_val = data[training_samples:training_samples + validation_samples]
y_val = labels[training_samples:training_samples + validation_samples]

"""解析GloVe词嵌入文件"""
glove_dir = 'F:\\IMDB数据集\\glove.6B'
embedding_index = {}
f = open(os.path.join(glove_dir,'glove.6B.100d.txt'),'r',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype='float32')
    embedding_index[word]=coefs
f.close()
print('Found %s word vectors.'%len(embedding_index))

"""准备GloVe词嵌入矩阵"""
embedding_dim = 100
embedding_matrix = np.zeros(shape=(max_words, embedding_dim))
for word,i in word_index.items():
    if i < max_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None: # 嵌入索引（embedding_index）中找不到词，其嵌入向量全为0
            embedding_matrix[i] = embedding_vector

"""定义模型"""
model = Sequential()
model.add(Embedding(max_words,embedding_dim,input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

"""在模型中加载GloVe嵌入(或不适用预训练)"""
model.layers[0].set_weights([embedding_matrix])# 与训练第一层（Embedding层）
model.layers[0].trainable = False # 冻结预训练层

"""编译并训练模型"""
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
model.save('pre_trained_glove_model.h5')

"""绘制结果"""
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)
plt.figure()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training(200) and Validation(10000) accuracy with pretrain')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Trainging(200) and Validation(10000) loss with pretrain')
plt.legend()

plt.show()