# 测试tensorflow效果
from __future__ import absolute_import, division, print_function, unicode_literals

# 安装 TensorFlow

import tensorflow as tf
mnist = tf.keras.datasets.mnist  # 导入数据集

# 使用data_load加载数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 构建简单模型
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# 编译
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练
model.fit(x_train, y_train, epochs=10)

# 验证
value = model.evaluate(x_test,  y_test, verbose=2)
print(value)

