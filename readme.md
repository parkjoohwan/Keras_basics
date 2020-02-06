# tensorflow 2.0 및 Keras 기본 튜토리얼

- OS : Windows 10
- Python Version : 3.7.5
- GPU : RTX 2060 SUPER 
- CUDA : 10.0
- TENSORFLOW : 2.0.0(현재 TF 최신버전은 2.1.X 2.1.X 사용시 쿠다 10.1 필요)


# 소스 별 참고 페이지
- [패션 이미지 분류(Basic_Image_Classification.py)](https://github.com/parkjoohwan/Keras_basics/blob/master/ML_Basic/Basic_Image_Classification.py) - [참고페이지](https://www.tensorflow.org/tutorials/keras/classification?hl=ko)
- [영화 리뷰 텍스트 분류(Text_Classification_with_TF_hub.py)](https://github.com/parkjoohwan/Keras_basics/blob/master/ML_Basic/Text_Classification_with_TF_hub.py) - [참고페이지](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub?hl=ko)
- [영화 리뷰를 사용한 텍스트 분류(Text_Classification_with_preprocessed_text.py)](https://github.com/parkjoohwan/Keras_basics/blob/master/ML_Basic/Text_Classification_with_preprocessed_text.py) - [참고페이지](https://www.tensorflow.org/tutorials/keras/text_classification?hl=ko)
- [자동차 연비 예측하기 : 회귀(Regression.py)](https://github.com/parkjoohwan/Keras_basics/blob/master/ML_Basic/Regression.py) - [참고페이지](https://www.tensorflow.org/tutorials/keras/regression?hl=ko)
- [과대적합과 과소적합(Overfit_And_Underfit.py)](https://github.com/parkjoohwan/Keras_basics/blob/master/ML_Basic/Overfit_And_Underfit.py) - [참고페이지](https://www.tensorflow.org/tutorials/keras/regression?hl=ko) 
- [모델 Save 및 Load(Save_and_Load.py)](https://github.com/parkjoohwan/Keras_basics/blob/master/ML_Basic/Save_and_Load.py) - [참고페이지](https://www.tensorflow.org/tutorials/keras/save_and_load?hl=ko)

- [CSV 파일 로드 및 생존 여부 예측(Load_CSV_Data.py)](https://github.com/parkjoohwan/Keras_basics/blob/master/Load_and_Preprocess_Data/Load_CSV_Data.py) - [참고페이지](https://www.tensorflow.org/tutorials/load_data/csv?hl=en)
- [npz 파일 로드 및 dataset 활용(Load_Numpy_Data.py)](https://github.com/parkjoohwan/Keras_basics/blob/master/Load_and_Preprocess_Data/Load_Numpy_Data.py) - [참고페이지](https://www.tensorflow.org/tutorials/load_data/numpy?hl=en)
- [pandas dataframe으로 dataset 활용(Load_Numpy_Data.py)](https://github.com/parkjoohwan/Keras_basics/blob/master/Load_and_Preprocess_Data/Load_Numpy_Data.py) - [참고페이지](https://www.tensorflow.org/tutorials/load_data/pandas_dataframe?hl=en)

- [Premade estimator를 이용한 홍채 분류(Premade_Estimator_Iris.py)](https://github.com/parkjoohwan/Keras_basics/blob/master/Estimator/Premade_Estimator_Iris.py) - [참고페이지](https://www.tensorflow.org/tutorials/estimator/premade?hl=en)
- [Estimators를 이용해 선형 모델 만들기(Linear_Model_With_Estimators.py)](https://github.com/parkjoohwan/Keras_basics/blob/master/Estimator/Linear_Model_With_Estimators.py) - [참고페이지](https://www.tensorflow.org/tutorials/estimator/linear?hl=en)
- [Estimators를 이용해 결정 트리 모델 만들기(Boosted_Trees_With_Estimators.py)](https://github.com/parkjoohwan/Keras_basics/blob/master/Estimator/Boosted_Trees_With_Estimators.py) - [참고페이지](https://www.tensorflow.org/tutorials/estimator/boosted_trees?hl=en)
# 필요 패키지 설치

```
- GPU를 사용할때만 requirements를 이용할것
pip install -r requirements_gpu.txt

- CPU를 이용할 경우는 tensorflow.org에서 확인 후 설치
혹은 requirements_gpu에서 gpu 버전만 제거 후 설치

```

`CPU의 경우에는 tensorflow, tf-nightly를 cpu 버전으로 설치해야함`