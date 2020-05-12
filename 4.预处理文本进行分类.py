# 使用tf.keras进行更新的操作

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries:{}, labels:{}".format(len(train_data),len(train_labels)))

print(train_data[0])

# a = len(train_labels[0])
# b = len(train_data[1])
# print(a, b)

# 创建一个辅助函数来查询一个包含了整数到字符串映射的字典对象
# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()

# 保留第一个索引
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(train_data[0]))

# 处理数据 因为我们在数据上要保证令电影评论的长度相同，因此可以是同pad_sequence来使数据标准化
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index["<PAD>"],
    padding = 'post',
    maxlen=256
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=256
)

print(len(train_data[0]))
print(train_data[0])

vocab_size = 10000
# 按照堆叠的方式构建分类器
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))  # 第一层使嵌入层。该层采用整数编码词汇表，并查找每个词索引的嵌入向量
# 这些向量是通过模型训练学习到的，向量向输出数组增加了一个维度。向量向输出数组增加了一个维度。
model.add(keras.layers.GlobalAveragePooling1D())
# GlobalAveragePooling1D 通过对序列维度求平均来为每个样本返回一个定长的输出向量。使模型可以处理变长数据
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 创建一个验证集
# 因为训练集和测试集已经在开发和调整模型的过程中使用过了。
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


# 训练模型

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=40,
    batch_size=512,   # 以512个样本的mini-batch大小，迭代训练模型。
    validation_data=(x_val, y_val),  # 在训练过程中，检测来自验证集的10000个样本上的损失值和精确度。
    verbose=1
)


# 验证模型精度 将返回两个值。损失值（loss）（一个表示误差的数字，值越低越好）与准确率（accuracy）。
results = model.evaluate(test_data, test_labels, verbose=2)
print(results)


# 创建一个准确率（accuracy）和损失值（loss）随时间变化的图表

# model.fit() 返回一个 History 对象，该对象包含一个字典，其中包含训练阶段所发生的一切事件：
history_dict = history.history
print(history_dict.keys())

import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)


# 绘制损失函数曲线
# "bo"代表'蓝点'
plt.plot(epochs, loss, 'bo', label='Training loss')
# b 代表‘蓝色直线’
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()

plt.show()


plt.clf()   # 清楚数字

# 绘制每个epoch精度的变化曲线
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()



