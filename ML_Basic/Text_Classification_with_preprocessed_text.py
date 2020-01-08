from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np

########################################################################################
#                                데이터 로드                                            #
########################################################################################

# IMDB 데이터셋 로드, 첫 실행시 다운로드, 이후는 캐시된 복사본 사용
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

"""
샘플 데이터 구조, 내용 확인

샘플 데이터는 단어를 매핑시킨 사전의 인덱스로 변환되어 저장되어있음

만약 'i like apple' 이라는 문장이 있고,

어휘 사전에 다음과 같이 매핑이 되어있다면

i -> 1
like -> 44
apple -> 56

i like apple -> [1, 44, 56] 같은 방식으로 변환 되어 있음
"""

print("학습 샘플 수: {}, 학습 라벨 수: {}".format(len(train_data), len(train_labels)))
print("첫 학습 샘플 내용: {}".format(train_data[0]))
print("첫 학습 샘플 단어 갯수: {}".format(len(train_data[0])))

# 단어와 정수 인덱스를 매핑한 딕셔너리
word_index = imdb.get_word_index()

# 처음 몇 개 인덱스는 사전에 정의되어 있습니다
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# 정수로 되어있는 데이터를 문자로 decode하는 함수
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# 함수 테스트
print("첫 샘플 데이터 문자로 보기 : {}".format(decode_review(train_data[0])))

########################################################################################
#                                데이터 전처리                                          #
########################################################################################

"""
리뷰-정수 배열-는 신경망에 주입하기 전에 텐서로 변환되어야 합니다. 변환하는 방법에는 몇 가지가 있습니다:

1. 원-핫 인코딩(one-hot encoding)은 정수 배열을 0과 1로 이루어진 벡터로 변환합니다. 
예를 들어 배열 [3, 5]을 인덱스 3과 5만 1이고 나머지는 모두 0인 10,000차원 벡터로 변환할 수 있습니다. 
그다음 실수 벡터 데이터를 다룰 수 있는 층-Dense 층-을 신경망의 첫 번째 층으로 사용합니다.
 이 방법은 num_words * num_reviews 크기의 행렬이 필요하기 때문에 메모리를 많이 사용합니다.
 
2. 다른 방법으로는, 정수 배열의 길이가 모두 같도록 패딩(padding)을 추가해 max_length * num_reviews 크기의 정수 텐서를 만듭니다.
 이런 형태의 텐서를 다룰 수 있는 임베딩(embedding) 층을 신경망의 첫 번째 층으로 사용할 수 있습니다.
 
이 튜토리얼에서는 두 번째 방식을 사용합니다.

영화 리뷰의 길이가 같아야 하므로 pad_sequences 함수를 사용해 길이를 맞추겠습니다:
"""

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print('샘플의 길이가 같아졌는지 확인\n첫번째 샘플 : {}\n임의의 샘플 : {}'.format(len(train_data[0]), len(train_data[555])))
print('샘플의 내용이 어떤지 확인\n첫번째 샘플:\n{}\n임의의 샘플:\n{}'.format(train_data[0], train_data[555]))

########################################################################################
#                                모델 구성                                              #
########################################################################################

"""
몇개의 layer를 사용할 것인가? => 4
각 layer에서 얼마나 많은 hidden unit을 사용할 것인가? => 16
예측의 결과는 무엇인가? => 긍정 : 1 부정 : 0 의 예측값
"""

# 입력 크기는 영화 리뷰 데이터셋에 적용된 어휘 사전의 크기입니다(10,000개의 단어)
vocab_size = 10000

model = keras.Sequential()
model.add(
    keras.layers.Embedding(vocab_size, 16, input_shape=(None,)))  # Embedding 층 output (batch, sequence, embedding)
model.add(keras.layers.GlobalAveragePooling1D())  # sequence 차원에 대해 평균을 계산, 고정된 길이의 출력벡터 반환
model.add(keras.layers.Dense(16, activation='relu'))  # 16개의 hidden unit을 가진 fc-layer
model.add(keras.layers.Dense(1, activation='sigmoid'))  # 출력

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 학습 데이터 중 1만개는 검증 데이터로 사용
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

# 라벨도 마찬가지
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

########################################################################################
#                                모델 평가                                              #
########################################################################################

results = model.evaluate(test_data, test_labels, verbose=2)

print(results)

history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

"""
    첫번째 그래프 
    
    점 - loss
    실선 - validation loss
    
    두번째 그래프
    
    점 - accuracy
    실선 - validation accuracy
"""

# loss / val_loss 그래프

# "bo"는 "파란색 점"입니다
plt.plot(epochs, loss, 'bo', label='Training loss')
# b는 "파란 실선"입니다
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()  # 그림을 초기화합니다

# accuracy / validation accuracy 그래프

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
