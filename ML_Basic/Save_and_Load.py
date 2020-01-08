from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras

########################################################################################
#                                데이터 로드                                            #
########################################################################################

# MNIST 데이터셋 활용 빠른 속도를 위해 1000개씩만
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


########################################################################################
#                                모델 구성                                             #
########################################################################################

# 간단한 Sequential 모델을 반환합니다
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# 모델 객체를 만듭니다
model = create_model()
model.summary()

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

"""
체크 포인트 콜백 사용, epoch 마다 체크포인트 파일을 업데이트
"""

# 체크포인트 콜백 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()

model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # 훈련 단계에 콜백을 전달합니다

# 옵티마이저의 상태를 저장하는 것과 관련되어 경고가 발생할 수 있습니다.
# 이 경고는 (그리고 이 노트북의 다른 비슷한 경고는) 이전 사용 방식을 권장하지 않기 위함이며 무시해도 좋습니다.


# 훈련되지 않은 모델만으로 테스트 셋 평가 해보기
model = create_model()

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("훈련되지 않은 모델의 정확도: {:5.2f}%".format(100 * acc))

# 체크포인트에서 가중치를 로드하고 테스트 셋 평가 해보기
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("복원된 모델의 정확도: {:5.2f}%".format(100 * acc))

# 파일 이름에 에포크 번호를 포함시킵니다(`str.format` 포맷)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # 다섯 번째 에포크마다 가중치를 저장합니다
    period=5)

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_images, train_labels,
          epochs=50, callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)

# 파일 이름에 에포크 번호를 포함시킵니다(`str.format` 포맷)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # 다섯 번째 에포크마다 가중치를 저장합니다
    period=5)

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_images, train_labels,
          epochs=50, callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

"""
Tensorflow는 기본적으로 최근 5개의 체크포인트만 저장함

체크포인트가 갖고있는 정보 
- 모델의 가중치를 포함하는 하나 이상의 shard
- 가중치가 어느 shard에 저장되어 있는지를 나타내는 인덱스 파일

가중치만 저장할 수도 있으나, 모델의 구조가 없으면 있으나 마나다.

물론 Tensorflow는 가중치만 저장, 모델의 구조만 저장, 전체 저장 모두 가능하다.
"""
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("복원된 모델의 정확도: {:5.2f}%".format(100 * acc))

# 가중치만을 저장합니다
model.save_weights('./checkpoints/my_checkpoint')

# 가중치를 복원합니다
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("복원된 모델의 정확도: {:5.2f}%".format(100 * acc))

model = create_model()

model.fit(train_images, train_labels, epochs=5)

# 전체 모델을 HDF5 파일로 저장합니다
model.save('my_model.h5')

# 가중치와 옵티마이저를 포함하여 정확히 동일한 모델을 다시 생성합니다
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print("복원된 모델의 정확도: {:5.2f}%".format(100 * acc))

model = create_model()

model.fit(train_images, train_labels, epochs=5)

# 타임 스탬프를 이름으로 가진 디렉토리에 저장하기 위해
import time

saved_model_path = "./saved_models/{}".format(int(time.time()))

tf.keras.experimental.export_saved_model(model, saved_model_path)
saved_model_path

new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
new_model.summary()

model.predict(test_images).shape

# 이 모델을 평가하려면 그전에 컴파일해야 합니다.
# 단지 저장된 모델의 배포라면 이 단계가 필요하지 않습니다.

new_model.compile(optimizer=model.optimizer,  # 복원된 옵티마이저를 사용합니다.
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# 복원된 모델을 평가합니다
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print("복원된 모델의 정확도: {:5.2f}%".format(100 * acc))
