from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
from IPython.display import clear_output
from matplotlib import pyplot as plt

# 타이타닉 데이터 불러오기
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
# 라벨은 마찬기자로 생존 여부
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

import tensorflow as tf

tf.random.set_seed(123)

print('------------------------데이터 확인 -----------------------\n{}'.format(dftrain.head()))
print('------------------------데이터 요약 -----------------------\n{}'.format(dftrain.describe()))

print('\n\n학습 데이터 갯수 : {}\n평가 데이터 갯수 : {}'.format(dftrain.shape[0], dfeval.shape[0]))



# 탑승객 나이 분포 히스토그램 확인
plt.subplot(3, 2, 1)
g1 = dftrain.age.hist(bins=20)
g1.set_title('populate age')

# 성별 분포 그래프 확인
plt.subplot(3, 2, 2)
g2 = dftrain.sex.value_counts().plot(kind='barh')
g2.set_title('populate sex')

# 탑승 클래스 분포 그래프 확인
plt.subplot(3, 2, 3)
g3 = dftrain['class'].value_counts().plot(kind='barh')
g3.set_title('populate class')

# 거주 지역 분포 그래프 확인
plt.subplot(3, 2, 4)
g4 = dftrain['embark_town'].value_counts().plot(kind='barh')
g4.set_title('populate embark_town')

# 성별에 따른 생존 비율 확인 그래프
plt.subplot(3, 2, 5)
g5 = pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh')
g5.set_xlabel('% survive')
g5.set_title('populate survive by sex')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

# 입력 feature column 생성

fc = tf.feature_column
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']


def one_hot_cat_column(feature_name, vocab):
    return tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(feature_name,
                                                                  vocab))


feature_columns = []
# 범주형
for feature_name in CATEGORICAL_COLUMNS:
    # 범주형 데이터들은 one-hot 인코딩
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(one_hot_cat_column(feature_name, vocabulary))
# 수치형
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name,
                                                            dtype=tf.float32))

example = dict(dftrain.head(1))
class_fc = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('class', ('First', 'Second', 'Third')))
print('Feature value: "{}"'.format(example['class'].iloc[0]))
print('One-hot encoded: ', tf.keras.layers.DenseFeatures([class_fc])(example).numpy())

print('\n\n입력 형식으로 변형된 feature 확인\n{}'.format(tf.keras.layers.DenseFeatures(feature_columns)(example).numpy()))

# 입력 함수 생성

# Use entire batch since this is such a small dataset.
NUM_EXAMPLES = len(y_train)


# 입력 함수
def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        # 학습을 위해 에폭만큼 반복
        dataset = dataset.repeat(n_epochs)
        # 메모리 학습 배치 사용 X
        dataset = dataset.batch(NUM_EXAMPLES)
        return dataset
    return input_fn


# 학습 및 평가 함수
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)


# boosted Trees model과 비교하기 위해 선형 분류기 학습

# 선형 분류 학습
linear_est = tf.estimator.LinearClassifier(feature_columns)

# 모델 훈련
linear_est.train(train_input_fn, max_steps=100)

# 평가
result = linear_est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))


# Boosted trees model 학습
# Regression과 Classification 둘 다 가능
# Since data fits into memory, use entire dataset per layer. It will be faster.
# Above one batch is defined as the entire dataset.
n_batches = 1
est = tf.estimator.BoostedTreesClassifier(feature_columns,
                                          n_batches_per_layer=n_batches)

# The model will stop training once the specified number of trees is built, not
# based on the number of steps.
# 단계수가 아닌 일 정 수 이상의 tree가 만들어지면 학습을 중단
est.train(train_input_fn, max_steps=100)

# Eval.
result = est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))


pred_dicts = list(est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')
plt.show()


from sklearn.metrics import roc_curve

# ROC 그래프 확인
fpr, tpr, _ = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0,)
plt.ylim(0,)
plt.show()