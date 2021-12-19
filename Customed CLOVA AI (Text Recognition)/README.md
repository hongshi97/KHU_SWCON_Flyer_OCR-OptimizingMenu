## Customed CLOVA AI (Text Recognition Model)
### 1. 개요
  1) 기존의 CLOVA AI Text Recognition Model은 영어로만 학습되어, 한국어의 Text Recognition에는 좋지 않은 성능 
  2) 이에, 한국어를 인식할 수 있는 모델 개발의 필요성을 느꼈으며, CLOVA AI가 제공하는 Pipeline을 기반으로 모델생성
  3) 0. 파일명에서 한글제거.ipynb ~ 4. train.ipynb 순으로 실행시 모델을 생성할 수 있음
  4) 활용데이터
     > 1) AIHUB([야외 실제 촬영 한글 이미지](https://aihub.or.kr/aidata/33985)): train 421,852장 / validation 52,733장
     > 2) extgenerator train 150,000장 / test 20,000장
  5) 모델을 훈련하기 위해서는 .lmdb 파일이 필요하며, 이를 구성하기 위해서 0~3단계의 ipynb 파일의 과정을 거침

### 2. 훈련을 위한 데이터 생성
  1) 0. 파일명에서 한글제거.ipynb
     > 1) AIHUB데이터의 파일명에 한글이 들어간 것을 제거
     > 2) opencv에서 원본이미지를 crop해야하는 작업이 필요한데, 이때 opencv는 파일명이 한글인 것을 읽어올 수 없기 때문
  2) 1. make_gtdata.ipynb
     > 1) .lmdb파일을 만들기 위해서는 gt_train(또는 gt_test).txt 파일과 원본 이미지 파일이 필요하다. 이때, gt_train(또는 gt_test).txt 파일을 만드는 과정
  3) 2. lmdb데이터 생성.ipynb
     > 1) gt_train.txt 파일과 train 원본 이미지를 통해 lmdb 데이터 생성
  4) 3. 데이터 내 모든 단어 추출.ipynb
     > 1) Text Recognition 모델 훈련 시 활용되는 데이터의 label을 지정하기 위함


### 3. 모델 훈련
  1) train.ipynb
     > 1) Text Recognition 모델 훈련
     > 2) Google Colab 환경, --character에 모델의 라벨을 대입
