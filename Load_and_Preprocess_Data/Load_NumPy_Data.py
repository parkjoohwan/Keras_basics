from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

"""
npz 파일에서 데이터를 어떻게 불러오고 이를 tf.data.Dataset에 넣는지에 대한 튜토리얼

이 코드에서는 MNIST dataset을 이용한다.
"""

# 데이터 로드
DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

# npz 파일을 읽어 train, test 셋으로 나눔
path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
with np.load(path) as data:
    train_examples = data['x_train']
    train_labels = data['y_train']
    test_examples = data['x_test']
    test_labels = data['y_test']

print('sample 확인 - shape : {}, data : {}'.format(train_examples[0].shape, train_examples[0]))


# tf.data.Dataset을 이용한 Numpy load
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              matrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.fit(train_dataset, epochs=10)

print('eavluate loss : {}'.format(model.evaluate(test_dataset)))