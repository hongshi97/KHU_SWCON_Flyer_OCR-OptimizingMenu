import os
import string
import argparse

import cv2 as cv
import numpy as np
import pandas as pd

import torch
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

import konlpy
from konlpy.utils import pprint
from konlpy.tag import Mecab, Kkma, Komoran, Hannanum, Okt

from recommender_modules.ocr_utils import *
from recommender_modules.optimization import *
from recommender_modules.recommender_utils import *

from gensim.models import word2vec
from PIL import Image 

####### 영양소 상하한 범위 도출 #######
def get_nutrient_boundary(dataset, sex, age):

    energy_lower = int(dataset.loc[(dataset['성별'] == sex) & (dataset['나이'] == age) & (dataset['영양성분'] == '에너지'), '하한값'])
    energy_upper = int(dataset.loc[(dataset['성별'] == sex) & (dataset['나이'] == age) & (dataset['영양성분'] == '에너지'), '상한값'])
    carbohydrate_lower = int(dataset.loc[(dataset['성별'] == sex) & (dataset['나이'] == age) & (dataset['영양성분'] == '탄수화물'), '하한값'])
    carbohydrate_upper = int(dataset.loc[(dataset['성별'] == sex) & (dataset['나이'] == age) & (dataset['영양성분'] == '탄수화물'), '상한값'])
    protein_lower = int(dataset.loc[(dataset['성별'] == sex) & (dataset['나이'] == age) & (dataset['영양성분'] == '단백질'), '하한값'])
    protein_upper = int(dataset.loc[(dataset['성별'] == sex) & (dataset['나이'] == age) & (dataset['영양성분'] == '단백질'), '상한값'])
    natrium_lower = int(dataset.loc[(dataset['성별'] == sex) & (dataset['나이'] == age) & (dataset['영양성분'] == '나트륨'), '하한값'])
    natrium_upper = int(dataset.loc[(dataset['성별'] == sex) & (dataset['나이'] == age) & (dataset['영양성분'] == '나트륨'), '상한값'])
    calcium_lower = int(dataset.loc[(dataset['성별'] == sex) & (dataset['나이'] == age) & (dataset['영양성분'] == '칼슘'), '하한값'])
    calcium_upper = int(dataset.loc[(dataset['성별'] == sex) & (dataset['나이'] == age) & (dataset['영양성분'] == '칼슘'), '상한값'])
    vitaminc_lower = int(dataset.loc[(dataset['성별'] == sex) & (dataset['나이'] == age) & (dataset['영양성분'] == '비타민c'), '하한값'])
    vitaminc_upper = int(dataset.loc[(dataset['성별'] == sex) & (dataset['나이'] == age) & (dataset['영양성분'] == '비타민c'), '상한값'])

    return [-energy_lower, -carbohydrate_lower, -protein_lower, -natrium_lower, -calcium_lower, -vitaminc_lower,
            energy_upper, carbohydrate_upper, protein_upper, natrium_upper, calcium_upper, vitaminc_upper]

####### 사용자 정보 log #######
def print_user_nutrient_range(nutrient_range, sex, age):

    energy_lower = -nutrient_range[0]
    energy_upper = nutrient_range[6]
    carbohydrate_lower = -nutrient_range[1]
    carbohydrate_upper = nutrient_range[7]
    protein_lower = -nutrient_range[2]
    protein_upper = nutrient_range[8]
    natrium_lower = -nutrient_range[3]
    natrium_upper = nutrient_range[9]
    calcium_lower = -nutrient_range[4]
    calcium_upper = nutrient_range[10]
    vitaminc_lower = -nutrient_range[5]
    vitaminc_upper = nutrient_range[11]

    print(f'\n==== 성별이 {sex}이고 나이가 {age}세인 당신의 1일 적정 영양성분 범위 ====')
    print(f'칼로리(kcal)의 적정범위: {energy_lower}kcal ~ {energy_upper}kcal')
    print(f'탄수화물(g)의 적정범위: {carbohydrate_lower}g ~ {carbohydrate_upper}g')
    print(f'단백질(g)의 적정범위: {protein_lower}g ~ {protein_upper}g')
    print(f'나트륨(mg)의 적정범위: {natrium_lower}mg ~ {natrium_upper}mg')
    print(f'칼슘(mg)의 적정범위: {calcium_lower}mg ~ {calcium_upper}mg')
    print(f'비타민c(mg)의 적정범위: {vitaminc_lower}mg ~ {vitaminc_upper}mg\n')