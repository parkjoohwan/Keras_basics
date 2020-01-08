from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import tensorflow as tf

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/applied-dl/heart.csv')
df = pd.read_csv(csv_file)

print('csv 데이터 확인 : \n\n{}'.format(df.head()))
print('data type 확인 : \n\n{}'.format(df.dtypes))

df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes

print('categorical 데이터 확인 : \n\n{}'.format(df.head()))

# tf.data.Dataset.from_tensor_slices 함수로 pandas dataframe 값을 읽어올 수 있다.
target = df.pop('target')

dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

# 5개의 feature target 출력
for feat, targ in dataset.take(5):
    print('Features : {}, Target : {}'.format(feat, targ))

print(tf.constant(df['thal']))

train_dataset = dataset.shuffle(len(df)).batch(1)

# 모델 설계 및 빌드
def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


model = get_compiled_model()
model.fit(train_dataset, epochs=15)

inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
x = tf.stack(list(inputs.values()), axis=-1)

x = tf.keras.layers.Dense(10, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model_func = tf.keras.Model(inputs=inputs, outputs=output)

model_func.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

# pandas의 dataframe의 열 구조를 유지하는 가장 쉬운 방법은 dictionary로 변환하고, 슬라이스 하는 것
dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values)).batch(16)

for dict_slice in dict_slices.take(1):
    print(dict_slice)

model_func.fit(dict_slices, epochs=15)






