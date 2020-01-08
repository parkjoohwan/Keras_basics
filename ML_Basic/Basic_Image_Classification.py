from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

# tensorflow와 tf.keras를 임포트합니다 tensorflow 2.0 이상만 가능
import tensorflow as tf
from tensorflow import keras

# 헬퍼(helper) 라이브러리를 임포트합니다
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__) tensorflow 버전 확인 (2.0)

########################################################################################
#                                데이터 로드                                            #
########################################################################################


# 패션 mnist 데이터 셋 로드
fashion_mnist = keras.datasets.fashion_mnist

"""
학습용, 테스트용 데이터 나누기
이미지는 28 x 28 크기의 넘파이, 픽셀 값은 0 ~ 255 (RGB) 
라벨은 0 ~ 9(옷의 종류) class_names 참고
"""

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('학습 데이터 구조 확인 : {}'.format(train_images.shape))
print('학습 데이터 라벨 갯수 확인 : {}'.format(len(train_labels)))
# print('학습 데이터 라벨 내용 확인 : {}'.format(train_labels))


print('테스트 데이터 구조 확인 : {}'.format(test_images.shape))
print('테스트 데이터 라벨 갯수 확인 : {}'.format(len(test_labels)))
# print('테스트 데이터 라벨 내용 확인 : {}'.format(test_labels))


########################################################################################
#                                데이터 전처리                                          #
########################################################################################

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# RGB의 값은 0~255 이를 0~1 사이의 값으로 정규화하기위해 255로 나눠준다.
# 학습, 테스트 데이터 동일하게 처리
train_images = train_images / 255.0
test_images = test_images / 255.0

########################################################################################
#                                 모델 구성                                            #
########################################################################################

# layer setting
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # 1차원 배열로 변환
    keras.layers.Dense(128, activation='relu'),  # fc-layer
    keras.layers.Dense(10, activation='softmax')  # 각 class에 대한 확률을 반환하며 총 합은 1
])

# model compile
# loss function - 모델의 오차 측정 함수
# optimizer - 데이터와 loss function을 바탕으로 모델의 업데이트 방법 결정
# metrics - 단계 모니터링을 위해 사용
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

########################################################################################
#                                 모델 훈련                                            #
########################################################################################

"""
    신경망 모델을 학습하는 과정
    1. 학습 데이터를 모델에 주입( train_images, train_labels)
    2. 모델이 이미지와 라벨을 매핑하는 방법을 학습
    3. 테스트 셋에 대한 예측을 통해 정확도를 확인
"""
# 학습 시작
model.fit(train_images, train_labels, epochs=5)

# 테스트 셋을 이용한 정확도 평가
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\n테스트 정확도:', test_acc)

# 학습된 모델로 테스트 이미지에 대한 예측 결과 만들기
predictions = model.predict(test_images)
print('\n첫 번째 테스트 데이터에 대한 예측 결과 : {}'.format(predictions[0]) +
      '\n첫 번째 테스트 데이터에 대한 예측 라벨 : {}'.format(class_names[np.argmax(predictions[0])]) +
      '\n첫 번째 테스트 데이터 실제 라벨 : {}'.format(class_names[test_labels[0]])
      )


# 클래스에 대한 예측을 그래프로 표현
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


# 그래프 값
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# 0번째 원소의 이미지, 예측, 정확도(신뢰도 점수) 확인
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

# 23번째 원소 확인
i = 22
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

# 처음 X 개의 테스트 이미지와 예측 레이블, 진짜 레이블을 출력합니다
# 올바른 예측은 파랑색으로 잘못된 예측은 빨강색으로 나타냅니다
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
plt.show()

# 테스트 세트에서 이미지 하나를 선택합니다
img = test_images[55]
# 이미지 하나만 사용할 때도 배치에 추가합니다
img = (np.expand_dims(img, 0))
predictions_single = model.predict(img)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

print('해당 이미지의 예측 라벨 : {} '.format(class_names[np.argmax(predictions_single[0])]))
print('해당 이미지의 라벨 : {} '.format(class_names[test_labels[55]]))
