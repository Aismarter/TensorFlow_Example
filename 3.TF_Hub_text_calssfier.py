
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf

import keras
import tensorflow_hub as hub
import tensorflow_datasets  as tfds
# 基于tensorflow-hub 和 tensorflow-datasets 的实验基本上失败 因为国内有墙
print("version:", tf.__version__)
print("Eager mode:", tf.executing_eagerly())
print("Hub version:", hub.__version__)
print("GPU is","available" if tf.config.experimental.list_physical_devices("CPU") else "NOT AVAILABLE")

# train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])
# tfds的数据有问题，可能是因为API参考出了问题。
# (train_data, validation_data), test_data = tfds.load(
#     name="imdb_reviews",
#     split=(train_validation_split, tfds.Split.TEST),
#     as_supervised=True
# )

# 参考tfdsapi文档修改的数据。
train_data, test_data = tfds.load(
    name="imdb_reviews",  # 暂时不写
    split=[
        tfds.core.ReadInstruction('train'),
        tfds.core.ReadInstruction('test'),
    ]   # 暂时安装训练集，验证集来分
)


