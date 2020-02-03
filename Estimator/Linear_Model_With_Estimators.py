from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

# 데이터 셋 로드
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # 학습용
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')   # 평가용
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

print("데이터셋 일부 확인:\n{}".format(dftrain.head()))
print("데이터셋 describe 확인:\n{}".format(dftrain.describe()))
print("라벨 확인:\n{}".format(y_train))

print("학습용 데이터셋 갯수 확인 : {}\n 평가용 데이터셋 갯수 확인: {}".format(dftrain.shape[0], dfeval.shape[0]))

# 학습용 데이터셋 나이 분포를 히스토그램으로 확인하기
dftrain.age.hist(bins=20)
plt.show()
# 성별 분포 확인
dftrain.sex.value_counts().plot(kind='barh')
plt.show()
# 탑승 클래스 분포 확인
dftrain['class'].value_counts().plot(kind='barh')
plt.show()
# 성별에 따른 생존 비율 확인
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
plt.show()


# 모델의 feature 엔지니어링
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
# 범주형 컬럼
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
# 수치형 컬럼
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# 입력 만드는 함수
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

ds = make_input_fn(dftrain, y_train, batch_size=10)()
for feature_batch, label_batch in ds.take(1):
  print('Some feature keys:', list(feature_batch.keys()))
  print()
  print('A batch of class:', feature_batch['class'].numpy())
  print()
  print('A batch of Labels:', label_batch.numpy())

age_column = feature_columns[7]
tf.keras.layers.DenseFeatures([age_column])(feature_batch).numpy()

gender_column = feature_columns[0]
tf.keras.layers.DenseFeatures([tf.feature_column.indicator_column(gender_column)])(feature_batch).numpy()

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result)

# 나이와 성별을에 대한 cross feature를 학습시키기 위해 이를 모델에 추가한다.
age_x_gender = tf.feature_column.crossed_column(['age', 'sex'], hash_bucket_size=100)

derived_feature_columns = [age_x_gender]
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns+derived_feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result)

pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')
plt.show()


from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

# ROC 그래프 확인하기기
fpr, tpr, _ = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0,)
plt.ylim(0,)
plt.show()


