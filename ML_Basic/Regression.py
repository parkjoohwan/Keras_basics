
"""
회귀(regression)는 가격이나 확률 같이 연속된 출력 값을 예측하는 목적
분류(classification)는 여러개의 클래스 중 하나의 클래스를 선택하는 목적
"""

from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

########################################################################################
#                                데이터 로드                                            #
########################################################################################

dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print('데이터 셋 위치 확인 : {}'.format(dataset_path))

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
print('데이터 셋 갯수 확인 : {}\n데이터 일부 확인 : \n{}'.format(len(dataset), dataset.tail()))


########################################################################################
#                                데이터 전처리                                          #
########################################################################################

"""
이 데이터셋은 일부 데이터가 누락되어 있는데, 이를 제거해야한다.    
"""

print('----- 항목 별 누락된 데이터 갯수 확인 -----\n{}'.format(dataset.isna().sum()))

dataset = dataset.dropna() # 데이터가 누락된 행 삭제

print("\n\n누락 된 행 제거 후 데이터 셋 갯수 확인 : {}\n\n".format(len(dataset)))

"""
    Origin 열은 수치를 나타내는게 아니기때문에 원 핫 인코딩으로 변환
"""

origin = dataset.pop('Origin')  # dataset 에서 Origin 열 추출

# Orgrin 값에 따라 해당 나라에 해당하면 1 아니면 0로 One-hot-Encoding

dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0

print("변화된 데이터 확인 : \n{}".format(dataset.tail()))

# 학습 셋(80%)과 테스트 셋(20)으로 분할
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# seaborn 패키지로 산점도 행렬을 만들어 확인하기

sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print("전반적 통계 확인 : \n{}".format(train_stats))

# 각 속성에서 타깃 값 또는 라벨을 분리하고, 이 라벨을 예측하기 위해 모델을 훈련 시킬 예정
# MPG = 연비
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# 데이터 정규화 함수
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


########################################################################################
#                                모델 구성                                             #
########################################################################################


# 모델 빌드 함수
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    return model

model = build_model()

model.summary()

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)

print('10개 샘플 결과 :\n{}'.format(example_result))

# 에포크가 끝날 때마다 점(.)을 출력해 훈련 진행 과정을 표시합니다
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

########################################################################################
#                                모델 평가                                             #
########################################################################################

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print("\n통계치\n{}".format(hist.tail()))

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(8,12))

  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.subplot(2,1,2)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

plot_history(history)

model = build_model()

# early_stop 콜백을 이용해서 성능 향상이 없으면 학습을 멈추게함
# patience 매개변수는 성능 향상을 체크할 에포크 횟수입니다
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("테스트 세트의 평균 절대 오차: {:5.2f} MPG".format(mae))

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")