from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras

print(tf.__version__)

# UCI机器学习库中包含很多机器学习开源数据集
dataset_path = keras.utils.get_file("auto-mpg.data","http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)


# 使用pandas导入数据集
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail())

# 对数据进行清洗
print(dataset.isna().sum())  # 先进行统计
dataset = dataset.dropna()  # 为了保证示例的简单性，删除这些行

origin = dataset.pop('Origin')  # ‘Origin’列实际上代表分类，而不仅仅是一个数字。所以把它转化成 独热码data

dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0

print(dataset.tail())


# 拆分数据集为训练数据和测试数据
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# 数据检查

# 快速查看训练集中几对列的联合分布
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')

# 查看总体数据的统计

train_stats = train_dataset.describe()
print(train_stats.pop('MPG'))
train_stats = train_stats.transpose()
print(train_stats)

# 从标签中分离特征
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# 数据规范化
# 对特征归一化是能提升模型综合性能的一种技巧。尽管模型可能未作归一化就收敛了，但实际上训练模型会变得复杂
# 并且生成的模型会依赖输入的单位长度

# 测试数据要与训练模型数据具有相同的分布


def norm(x):
    return (x - train_stats['mean']) /train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


def build_model():
    model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

model = build_model()
model.summary()   # 输出模型的简单描述

# 从训练数据中批量获取10条例子，并对这几个例子调用 model.predict
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)



# 训练模型，对模型进行1000个周期的训练，并在history对象中记录训练和验证的准确性
# 通过为每一个完成的时期打印一个点来显示训练进度
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs):
        if epoch % 100 == 0:print('')
        print('.', end='')


EPOCHS = 200

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[PrintDot()])

# 使用history对象中存储的统计信息可视化模型的训练进度
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(
        hist['epoch'],hist['mae'],
        label='Train Error')
    plt.plot(hist['epoch'],hist['val_mae'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['mse'], label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()

plot_history(history)


# 若在训练100个epochs后误差非但没有改进反而出现恶化。
# 更新model.fit调用，使用EarlyStopping Callback策略来优化算法
model = build_model()

# patience值用来检查改进epochs的数量
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])
plot_history(history)

loss, mae, mse = model.evaluate(normed_train_data, test_labels, verbose=2)
print("Testing set Mean Abs Error:{:5.2f} MPG".format(mae))

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels,test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

error = test_predictions - test_labels
plt.hist(error, bins= 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")

