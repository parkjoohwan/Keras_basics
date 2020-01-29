from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf

import pandas as pd

######################################################################################################
#                                        데이터 세팅                                                  #
######################################################################################################

# CSV 컬럼지정
# 홍채 데이터 셋은 4가지 feature과 1개의 label이 있음
# * Feature
# sepal length
# sepal width
# petal length
# petal width
# * Label
# Species

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
# Species 구분
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

print("학습 데이터 일부 확인\n{}".format(train.head()))

# 라벨 추출
train_y = train.pop('Species')
test_y = test.pop('Species')

# 이후 라벨을 제외한 dataframe 확인 가능
print("학습 데이터 일부 확인\n{}".format(train.head()))

# ** 사전 작성된 Estimator를 기반으로 TF 프로그램을 구현하려면, 아래의 항목을 수행해야함
# 1개 이상의 input function 구현
# model의 feature column을 정의
# feature column과 다양한 hyper parameter를 지정해서 esimator를 인스턴스화
# 적절한 input function을 데이터 소스로 전달해서 estimator 오브젝트에서 1개 이상의 함수 호출


# input function 구현
# input function은 feature과 label 튜플을 출력하는 tf.data.Dataset 객체 반환
# 아래 주석은 예제
############################ example ############################
# def input_evaluation_set():
#     features = {'SepalLength': np.array([6.4, 5.0]),
#                 'SepalWidth':  np.array([2.8, 2.3]),
#                 'PetalLength': np.array([5.6, 3.3]),
#                 'PetalWidth':  np.array([2.2, 1.0])}
#     labels = np.array([2, 1])
#     return features, labels
#################################################################

# DataSet API 이용, 모든 종류의 데이터 구문분석 용이

def input_fn(features, labels, training=True, batch_size=256):
    """An input function for training or evaluating"""
    # input을 dataset으로 변환
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # 학습 모드일경우 셔플 및 반복
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

# feature 컬럼 정의
# 홍채 데이터 셋에서 feature column은 4개고, 이 4개의 column은 numeric 데이터이기 때문에, 이를 부동소숫점으로 지정해줘야함

# Feature columns describe how to use the input.
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# estimator 인스턴스화

# 홍채 분류 문제는 TF에서 몇개의 pre-made classifier estimator를 지원함
# tf.estimator.DNNClassifier - 멀티 클래스를 갖는 분류를 위한 심층 모델을 위한 estimator
# tf.estimaotr.DNNLinearCombinedClassifier - 넓고 깊은 모델을 위한 estimator
# tf.estimator.LinearClassifier - 선형 모델을 기반으로하는 분류기를 위한 estimator

# 이 코드에서는 tf.estimator.DNNClassifier가 최선일 것 같으므로 이를 사용
# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # 모델은 반드시 3개의 class 중에 하나를 반환해야함
    n_classes=3)

######################################################################################################
#                                   학습, 평가, 예측                                                  #
######################################################################################################

# Train the Model.
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)

# 평가에는 steps가 선언되지않음, 1개의 epoch만으로 평가
eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\n테스트 셋에 대한 정확도: {accuracy:0.3f}\n'.format(**eval_result))

# Generate predictions from the model
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}

def input_fn(features, batch_size=256):
    """An input function for prediction."""
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

# predict 함수는 python iterable을 반환
predictions = classifier.predict(
    input_fn=lambda: input_fn(predict_x))


for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('예측은 "{}" ({:.1f}%), 실제는 "{}"'.format(
        SPECIES[class_id], 100 * probability, expec))

