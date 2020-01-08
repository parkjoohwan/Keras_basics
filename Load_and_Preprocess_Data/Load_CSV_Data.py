from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import pandas as pd
import numpy as np
import tensorflow as tf

"""
CSV 파일에서 데이터를 어떻게 불러오고 이를 tf.data.Dataset에 넣는지에 대한 튜토리얼

데이터는 타이타닉 승객 리스트이다.
"""

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

print('CSV 파일 위치 : {} , {}'.format(train_file_path, test_file_path))

"""
위 파일 위치의 학습 데이터를 head train.csv로 보면 다음과 같은 구조임을 확인 할 수 있음
----------------------------------------------------------------------------
survived,sex,age,n_siblings_spouses,parch,fare,class,deck,embark_town,alone
0,male,22.0,1,0,7.25,Third,unknown,Southampton,n
1,female,38.0,1,0,71.2833,First,C,Cherbourg,n
1,female,26.0,0,0,7.925,Third,unknown,Southampton,y
1,female,35.0,1,0,53.1,First,C,Southampton,n
0,male,28.0,0,0,8.4583,Third,unknown,Queenstown,y
0,male,2.0,3,1,21.075,Third,unknown,Southampton,n
1,female,27.0,0,2,11.1333,Third,unknown,Southampton,n
1,female,14.0,1,0,30.0708,Second,unknown,Cherbourg,n
1,female,4.0,1,1,16.7,Third,G,Southampton,n
----------------------------------------------------------------------------

10개의 열로 구성되어있고 구분자는 ,인 것을 확인할 수 있다. 
"""

# pandas 를 이용해 확인

column_names = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch',
                'fare', 'class', 'deck', 'embark_town', 'alone']
raw_dataset = pd.read_csv(train_file_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=",", skipinitialspace=True)

dataset = raw_dataset.copy()
print('데이터 셋 갯수 확인 : {}\n데이터 일부 확인 : \n{}'.format(len(dataset), dataset.tail()))

LABEL_COLUMN = 'survived'
LABELS = [0, 1]


# csv 파일을 읽어 dataset으로 가져오는 함수
def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=5,  # Artificially small to make examples easier to show.
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
        **kwargs)
    return dataset


raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)


# dataset 확인 함수
def show_batch(dataset):
    print('-------------------------------------------------------------------')
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))


show_batch(raw_train_data)
show_batch(raw_test_data)

# 모든 컬럼 조회
print('모든 컬럼 조회하기')

CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']
temp_dataset = get_dataset(train_file_path, column_names=CSV_COLUMNS)
show_batch(temp_dataset)

# 특정 컬럼만 조회하기
print('특정 컬럼만 조회하기')

SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'class', 'deck', 'alone']
temp_dataset = get_dataset(train_file_path, select_columns=SELECT_COLUMNS)
show_batch(temp_dataset)

# 컬럼 타입을 원하는데로 지정하기 ( numeric )
print('숫자 타입의 형식을 원하는데로 지정하기')
SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'parch', 'fare']
DEFAULTS = [0, 0.0, 0.0, 0.0, 0.0]
temp_dataset = get_dataset(train_file_path,
                           select_columns=SELECT_COLUMNS,
                           column_defaults=DEFAULTS)
show_batch(temp_dataset)

example_batch, labels_batch = next(iter(temp_dataset))


def pack(features, label):
    return tf.stack(list(features.values()), axis=-1), label


packed_dataset = temp_dataset.map(pack)
print('dataset pack : {}'.format(packed_dataset))

for features, labels in packed_dataset.take(1):
    print(features.numpy())
    print()
    print(labels.numpy())

show_batch(raw_train_data)
example_batch, labels_batch = next(iter(temp_dataset))


# 숫자 형태의 Features Pack
class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features, labels


NUMERIC_FEATURES = ['age', 'n_siblings_spouses', 'parch', 'fare']

packed_train_data = raw_train_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))

packed_test_data = raw_test_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))

show_batch(packed_train_data)

# Data 정규화
example_batch, labels_batch = next(iter(packed_train_data))

desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()
print(desc)

MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])


def normalize_numeric_data(data, mean, std):
    # Center the data
    return (data - mean) / std


# See what you just created.
normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]
print('\nnumeric_cloumn : {}'.format(numeric_column))

print('\n샘플 배치\n{}'.format(example_batch['numeric']))


# 카데고리적인 Data ex) 성별 : 남/여 변환
CATEGORIES = {
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
}

categorical_columns = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))

# See what you just created.
print('categorical_columns : {}'.format(categorical_columns))

categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
print('\n\nexample_batch : {}'.format(example_batch))
print(categorical_layer(example_batch).numpy()[0])

# 전처리 layer
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)
print(preprocessing_layer(example_batch).numpy()[0])

# model 컴파일
model = tf.keras.Sequential([
  preprocessing_layer,
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# 학습
train_data = packed_train_data.shuffle(500)
test_data = packed_test_data

model.fit(train_data, epochs=20)

# 평가
test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

# 생존 여부 예측
predictions = model.predict(test_data)

# Show some results
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
  print("Predicted survival: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"))
