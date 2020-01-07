from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("허브 버전: ", hub.__version__)
print("GPU ", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "사용 불가능")

########################################################################################
#                                데이터 로드                                            #
########################################################################################

# IMDB 리뷰 데이터셋을 불러와 학습, 테스트 데이터로 나눕니다.
# 훈련 세트를 6대 4로 나눕니다.
# 결국 훈련에 15,000개 샘플, 검증에 10,000개 샘플, 테스트에 25,000개 샘플을 사용하게 됩니다.
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews",
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True)

# 첫 10개의 데이터/라벨 보기
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print('data 형태 및 내용 : {}'.format(train_examples_batch))
print('data label 형태 및 내용 : {}'.format(train_labels_batch))

########################################################################################
#                                 모델 구성                                            #
########################################################################################

"""
입력 데이터는 text, 예측 값은 긍정/부정(0,1)

텍스트를 임베딩 벡터로 변환, 첫번째 layer로 pre-trained 된 텍스트 임베딩 사용
"""

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

model = tf.keras.Sequential()
model.add(hub_layer)    # 텍스트를 임베딩 벡터로 매핑하는 pre-trained된 tensorflow hub 모델 output - (num_examples, embedding_dimension)
model.add(tf.keras.layers.Dense(16, activation='relu')) # hidden layer
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))   # 1개의 출력, 0~1 사이의 값

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

########################################################################################
#                                 모델 훈련                                            #
########################################################################################

history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

########################################################################################
#                                 모델 평가                                            #
########################################################################################

results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))