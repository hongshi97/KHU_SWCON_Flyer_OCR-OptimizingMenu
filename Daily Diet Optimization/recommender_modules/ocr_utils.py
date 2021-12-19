import os
import string
import argparse

import cv2 as cv
import numpy as np
import pandas as pd

import OCR_modules.demo

import torch
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

import konlpy
from konlpy.utils import pprint
from konlpy.tag import Mecab, Kkma, Komoran, Hannanum, Okt

from recommender_modules.etc_utils import *
from recommender_modules.optimization import *
from recommender_modules.recommender_utils import *

from gensim.models import word2vec
from PIL import Image

######## Text Detection에서 Recognition까지 pipeline ########

def get_OCR_result(leaflet_filename, ingred_list, DETECTION_MODEL, MECAB):

    ############ 1. Text detection ############
    ### Load image
    image_path = f"leaflet/{leaflet_filename}.jpg"
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    leaflet_image = DocumentFile.from_images(image_path)
    original_axis_x, original_axis_y = leaflet_image[0].shape[1], leaflet_image[0].shape[0]

    ### data to Model
    crop_result = DETECTION_MODEL(leaflet_image)
    json_output = crop_result.export()

    ### 좌표 구하기
    cnt = 0
    for blocks in json_output['pages'][0]['blocks']:
    
        lines = blocks['lines']
        for line in lines:
    
            ### 좌표도출
            bounding_box = line['words'][0]
            x1 = int(bounding_box['geometry'][0][0] * original_axis_x)
            y1 = int(bounding_box['geometry'][0][1] * original_axis_y)
            x2 = int(bounding_box['geometry'][1][0] * original_axis_x)
            y2 = int(bounding_box['geometry'][1][1] * original_axis_y)
    
            ### 그림그리기
            img_cropped = image[y1:y2, x1:x2, :]
            cv.imwrite(f'contour_output/{leaflet_filename}/{leaflet_filename}_sample{cnt+1}.jpg', img_cropped)
    
            cnt += 1


    ############ 2. Text Recognition ############
    test_opt = OCR_modules.demo.options
    test_opt.image_folder = f'contour_output/{leaflet_filename}/' # contour output이 저장되는 폴더명
    test_opt.saved_model = 'model_weight/best_accuracy.pth' # Model weight 정의
    test_opt.workers = 0

    final_result = OCR_modules.demo.Prediction(test_opt)


    ############ 3. Get Ingredient Name in Leaflet ############

    ### mecab() 결과를 list로 변환 후 다시 set로 변환 => 교집합 구하기 위해 set 형태로.
    final_result['mecab_output'] = final_result['outputs'].apply(MECAB.nouns)  # OCR 결과에 mecab 적용해서 명사 추출
    mecab_output = final_result.mecab_output.tolist()   # 위 라인에서 추출된 식재료명 명사들을 List 형태로 저장

    words = []
    for word_list in mecab_output:
        for word in word_list:
            words.append(word)

    words_set = set(words)  # 최종적으로 Set 형태로 저장


    ### 잡다한 단어 제거
    one_word_remove_list = ['중', '삶', '진', '고', '펜', '알', '뼈', '청', '캔', '체']
    two_word_remove_list = ['무절', '양념', '각종', '콩비', '큐브', '장식', '토핑', '밑간', '메스', '청소', '옐로', '레드', '아삭', '양차', '천연', '소스', '분당', '재료', '반찬']
    remove_list = one_word_remove_list + two_word_remove_list

    for remove_word in remove_list:
        if remove_word in ingred_list:
            ingred_list.remove(remove_word)

    ingre_dict = set(ingred_list)

    # "식재료데이터 사전"(ingre_dict)과 "전단지에서 추출한 명사"(words_set) 교집합 구하기 => 최종 전단지 OCR 결과 식재료명 리스트(ocr_outputs)
    ocr_outputs = list(ingre_dict.intersection(words_set))

    return ingre_dict, ocr_outputs