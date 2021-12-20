## Customed CLOVA AI (Text Recognition Model)
### 1. 개요
  1) 기존의 CLOVA AI Text Recognition Model은 영어로만 학습되어, 한국어의 Text Recognition에는 좋지 않은 성능 
  2) 이에, 한국어를 인식할 수 있는 모델 개발의 필요성을 느꼈으며, CLOVA AI가 제공하는 Pipeline을 기반으로 모델생성
  3) 파일명에서 한글제거.ipynb ~ train.ipynb 순으로 실행시 모델을 생성할 수 있음
  4) 활용데이터

         (1) AIHUB([야외 실제 촬영 한글 이미지](https://aihub.or.kr/aidata/33985)): train 421,852장 / validation 52,733장
          - 이때, 세로형 간판 이미지는 제거
         (2) extgenerator train 150,000장 / test 20,000장
         
  5) 모델을 훈련하기 위해서는 .lmdb 파일이 필요하며, 이를 구성하기 위해서 0~3단계의 ipynb 파일의 과정을 거침

### 2. 훈련을 위한 데이터 생성
  1) 파일명에서 한글제거.ipynb

         (1) AIHUB데이터의 파일명에 한글이 들어간 것을 제거
         (2) opencv에서 원본이미지를 crop해야하는 작업이 필요한데, 이때 opencv는 파일명이 한글인 것을 읽어올 수 없기 때문
          
  2) make_gtdata.ipynb

         (1) .lmdb파일을 만들기 위해서는 gt_train(또는 gt_test).txt 파일과 원본 이미지 파일이 필요
         (2) 이때, gt_train(또는 gt_test).txt 파일을 만드는 과정
          
  3) lmdb데이터 생성.ipynb

         gt_train.txt 파일과 train 원본 이미지를 통해 lmdb 데이터 생성
          
  4) 데이터 내 모든 단어 추출.ipynb

         Text Recognition 모델 훈련 시 활용되는 데이터의 label을 지정하기 위함


### 3. 모델 훈련
  1) train.ipynb

         Text Recognition 모델 훈련
         Google Colab 환경, --character에 모델의 라벨을 대입
     
     
### 4. 기타 폴더 및 파일 설명
  1) .py 파일(create_lmdb_dataset.py ~ utils.py) 및 modules 폴더(feature_extraction.py ~ transformation.py)
  
          Text Recognition 모델 훈련시 필요한 모듈들
          
  2) data 폴더

          (1) 개요: 훈련에 쓰일 AIHUB를 전처리하고 gt.txt 데이터까지 만드는 과정
          (2) test 폴더 / train 폴더: AIHUB의 원본이미지를 Crop한 이미지데이터 경로
          (3) test_image폴더 / train_image 폴더: AIHUB의 원본이미지 경로
          (4) test_label 폴더 / train_label 폴더: AIHUB의 원본라벨 경로
          (5) gt_train.txt / gt_test.txt: 각 image에 대응하는 label값을 명시한 파일

  3) data_lmdb 폴더

          (1) 각 train과 test의 .lmdb데이터의 저장경로
          (2) .lmdb는 모델 훈련을 위한 데이터의 형태
          (3) 데이터는 각 폴더에 첨부된 링크에서 다운로드

  4) textimagegenerator 폴더

          (1) data 폴더 
           - test 폴더 / train 폴더: generator에서 생성된 이미지들이 저장되는 경로
           - test_label 폴더 / train_label 폴더: generator에서 생성된 이미지들의 파일명과 라벨을 명시한 파일
           
          (2) fonts 폴더
           - generator 시 활용하는 여러 글꼴 데이터 경로
           
          (3) sample_data.xlsx
           - generator 시 활용하는 랜덤 문구 (뉴스기사 데이터)

          (4) TextImageGenerator.ipynb
           - Text image를 generator하는 파일
