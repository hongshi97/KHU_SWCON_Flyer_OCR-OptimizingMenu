# <딥러닝 기반의 대형마트 전단지 인식을 통한 주간메뉴 최적화> 

## Daily Diet Optimization 
 ### Process (Daily Diet Optimization.ipynb 파일 참고)
   1) 전단지 OCR(Text detection + Text Recognition)
 
     > Text Detection: DocTr pretrained model
     > Text Recognition: Customed CLOVA AI Text Recognition (한국어 훈련)
 
   2) Association Rule Discovery의 Support기반 궁합이 좋은 식재료 선별 
   3) 최적화(이진 정수계획법) 기반의 1일 식단메뉴 구성

 ### 참고사항
   1) **Daily Diet Optimization/dataset**: 2020년 한국인 1일 영양섭취기준 데이터, 레시피 데이터(출처 해먹남녀)의 데이터 경로
   2) **Daily Diet Optimization/leaflet**: 분석하고자 하는 전단지의 저장 경로
   3) **Daily Diet Optimization/OCR_modules**: "1) 전단지 OCR(Text detection + Text Recognition)"을 활용하기 위한 모듈
   4) **Daily Diet Optimization/recommender_modules**: Association Rule Discovery의 Support기반 궁합이 좋은 식재료 선별과 최적화 기반의 1일 식단메뉴 구성을 활용하기 위한 모듈

 ### 주의사항
   1) Google Colab 기준으로 작성하였음
   2) Daily Diet Optimization/model_weight: 모델의 weight file은 아래 링크에서 다운로드 후 폴더에 업로드
   3) Install packages 부분을 실행 → 런타임 다시시작 → Set Environment부터 시작

 

## 
