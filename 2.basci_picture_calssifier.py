# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import sys


print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# 加载数据
class_names = ['T-shrit/top','Trouser','Pullover','Dress','Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 显示一张图片
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

print(train_images.shape)
# 显示数据
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# 构建模型 含有一个隐含层的全连接层
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    # 这一步的作用只是将28*28的二维图像数据，转化成784的一维数据
    keras.layers.Dense(128, activation='relu'),  # 包含128个节点的隐藏层
    keras.layers.Dense(10)  # 包含10个节点的输出层
])

# 编译模型
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 训练模型
model.fit(train_images, train_labels, epochs=10)


# 测试模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest Accuracy:', test_acc)

# 加入全连接层进行分类
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# 对图片进行分类
predictions = probability_model.predict(test_images)
# 输出对不同种类的预测结果
print(predictions[0])
# 输出最大结果
print('最大可能的结果:',class_names[np.argmax(predictions[0])])

# 构建输出的结果图片，同时输出底部的可能值
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
    predict_label = np.argmax(predictions_array)
    if predict_label == true_label:
        color = "blue"
    else:
        color = 'red'
    # 在图片底部展示输出结果
    plt.xlabel("{} {:2.0f}%({})".format(class_names[predict_label],
                                        100*np.max(predictions_array),
                                        class_names[true_label]),
                                        color=color)

# 输出每一类的预测结果
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predict_label = np.argmax(predictions_array)

    thisplot[predict_label].set_color('red')
    thisplot[true_label].set_color('blue')

# 图片显示一张测试集的预测结果
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# 图片展示另一张测试样本的预测结果
i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# Plot the frist X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.

# 联合展示15张图片的预测结果
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# 添加新图片进行预测
# add a picture
# Grab an image from the test dataset.
img = test_images[1]
print(img.shape)

img = (np.expand_dims(img,0))
print(img.shape)

prediction_single = probability_model.predict(img)
print(prediction_single)

plot_value_array(1, prediction_single[0],test_labels)
_=plt.xticks(range(10), class_names, rotation=45)
plt.show()

print(np.argmax(prediction_single[0]))


