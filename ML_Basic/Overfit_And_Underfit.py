from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

def multi_hot_sequences(sequences, dimension):
    # 0으로 채워진 (len(sequences), dimension) 크기의 행렬을 만듭니다
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # results[i]의 특정 인덱스만 1로 설정합니다
    return results


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

plt.plot(train_data[0])
plt.show()

"""
Overfitting을 막는 가장 간단한 방법은 모델의 규모를 축소하는 것

즉, 모델에 있는 학습 가능한 파라미터의 수를 줄이는것
(모델 파라미터는 layer의 갯수와 layer의 unit 개수에 의해 결정됨)

딥러닝에서는 모델의 학습 가능한 파라미터의 수를 모델의 '용량'이라고 말하기도 함

직관적으로 생각해보면 많은 파라미터를 가진 모델이 더 많은 '기억 용량'을 가지지만,
이런 모델은 학습 샘플과 타깃 사이를 딕셔너리와 같은 매핑으로 완벽하게 학습 가능하다 
하지만 이전에 본 적 없는(학습시키지 않은) 데이터를 예측할 땐 쓸모 없을 것이다.

가장 중요한 것은 [일반화(Normalization)]

반면 네트워크의 기억 용량이 부족하다면 이런 매핑을 쉽게 학습할 수 없을 것이다. 
손실을 최소화 하기 위해선 예측 성능이 더 많은 압축된 표현을 학습해야하지만
그렇다고 너무 작은 모델을 만들면 훈련 데이터를 학습하기 어렵게된다.

그래서 이 Overfit과 Underfit 사이의 균형을 잘 잡아야한다.

그리고 이를 잘 잡기 위한 완벽한 공식, 모델 구조는 없다. 여러가지 다른 구조를 실험하며 
찾아내야만 한다.

알맞은 모델의 크기를 찾기 위해선 적은 수의 layer와 파라미터로 시작해서 
점차 validation loss가 감소할때까지 새로운 layer를  추가하거나 크기를 늘리는 것이 좋다.

이 코드에서는 Dense 층만 사용하는 간단한 기준 모델을 만들고,
작은 규모의 모델과 큰 버전의 모델을 만들어 비교한다.
"""


########################################################################################
#                                모델 구성                                             #
########################################################################################

# 기준 모델
baseline_model = keras.Sequential([
    # `.summary` 메서드 때문에 `input_shape`가 필요합니다
    keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])

baseline_model.summary()

baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)

# 작은 모델
smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

smaller_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])

smaller_model.summary()

# 같은 데이터로 학습
smaller_history = smaller_model.fit(train_data,
                                    train_labels,
                                    epochs=20,
                                    batch_size=512,
                                    validation_data=(test_data, test_labels),
                                    verbose=2)

# 큰 모델
bigger_model = keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

bigger_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy','binary_crossentropy'])

bigger_model.summary()

# 마찬가지로 같은 데이터 활용
bigger_history = bigger_model.fit(train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)

# 학습 과정 그래프 실선 : loss,  점선 : validation loss
def plot_history(histories, key='binary_crossentropy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])
  plt.show()


plot_history([('baseline', baseline_history),
              ('smaller', smaller_history),
              ('bigger', bigger_history)])

"""
Overfitting을 방지하기 위한 전략

- 더 많은 학습 데이터 수집
- 네트워크의 용량을 줄이기
- 가중치 규제
- 드롭아웃 추가

entropy과 같은 작은 모델(적은 파라미터를 가진 모델)은 Overfitting 되는 일이 작을 것이다.
따라서 Overfitting을 완화시키는 가장 일반적인 방법은 가중치가 작은 값을 가지도록 네트워크의 
복잡도에 제약을 가한다. 이는 가중치 값의 분포를 좀 더 균일하게 만듬 ( 가중치 규제 )
네트워크의 loss function에 큰 가중치에 해당하는 비용을 추가. 

1. L1 규제 > 가중치의 절대값에 비례하는 비용 추가 ( 가중치의 L1 nrom )
2. L2 규제 > 가중치의 제곱에 비례한느 비용 추가 ( 가중치의 L2 nrom^2 ) 이를 가중치 감쇠라고 부름

L1 규제는 일부 가중치 파라미터를 0으로 만든다.
그에 반에 L2 규제는 가중치 파라미터를 제한하지만 완전히 0으로 만들지는 않는다.
"""

# L2 규제 모델
l2_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

l2_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', 'binary_crossentropy'])

l2_model_history = l2_model.fit(train_data, train_labels,
                                epochs=20,
                                batch_size=512,
                                validation_data=(test_data, test_labels),
                                verbose=2)

plot_history([('baseline', baseline_history),
              ('l2', l2_model_history)])


"""
신경망에서 가장 효과적이고 널리 사용하는 규제 기법은 dropout이다.
이 dropout을 layer에 적용하면 훈련하는 동안 층의 출력 특성을 랜덤하게 0으로 만든다.
보통 0.2 ~ 0.5 사이의 dropout 비율을 주는데, 이는 0이 되는 특성의 비율이다.

단, 테스트 단계에서는 dropout을 적용하지 않는다.
"""


# dropout을 적용한 모델

dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

dpt_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy','binary_crossentropy'])

dpt_model_history = dpt_model.fit(train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)

plot_history([('baseline', baseline_history),
              ('dropout', dpt_model_history)])

