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

from recommender_modules.etc_utils import *
from recommender_modules.ocr_utils import *
from recommender_modules.optimization import *

from gensim.models import word2vec
from itertools import combinations
from PIL import Image 

########### Support값 계산 ###########
def support(word1,word2, recipes):

    num1=0
    num2=0
    both=0

    for i in recipes:
        if word1 in i and word2 in i:
            both+=1

    ret=round(both/5219,6)

    return ret


########### 적절한 재료 조합을 Association Rule Discovery 기반으로 산출 ###########
def ingred_combination_asr(ingredient_dict, user_input, ocr_output, recipes, topn = 10):


    # input 정의
    total_input = user_input + ocr_output
    ingred_combinations = list(combinations(total_input, 2))
    combinations_dict = {}
    temp_list = ingred_combinations


    # 돼지고기, 소고기, 닭고기, 기본 조미료 정의 및 Set 형태로 정의
    pig=['돼지고기뒷다리살', '돼지고기대패목심','돼지고기안심','돼지고기갈비','돼지고기삼겹살','돼지고기카레용','돼지비계','돼지등갈비','돼지고기목살','돼지족발','대패삼겹살','돼지껍데기','채썬돼지고기','삼겹살','돼지고기볶음용','돼지고기앞다릿살','돼지등뼈','돼지고기불고기용','돼지고기','돼지고기등심','돼지고기국물용','돼지고기수육용','돼지사골육수','돼지고기다짐육']
    cow=['소고기채끝살','소고기국물용','소고기양지','소고기항정살','소고기볶음고추장','소고기장조림','소고기부채살','채썬소고기','소고기우둔살','소고기장조림용','불고기감소고기','소고기불고기용','소고기','소고기다짐육','소고기안심','소고기홍두깨살','소고기스테이크용','소고기큐브','소고기한우양지','소고기채끝등심','소고기샤브샤브용','소고기등심','소고기목심','소고기한우채끝','소고기한우','차돌박이','소고기육회용']
    chicken=['닭다리살','닭고기통조림','양념치킨','닭모래집','닭안심살','후라이드치킨','훈제닭가슴살','닭고기포','치킨텐더','닭근위','닭고기넓적다리살','치킨까스','닭발','닭다리','볼케이노치킨','닭날개','닭갈비','닭고기','무뼈닭발','훈제닭다리','닭껍질','치킨','닭똥집','치킨너겟','닭고기볶음탕용','닭강정','냉동무뼈닭발','닭고기안심','닭가슴살','닭봉']
    etc_meet=['불고기','고기','고기다짐육','갈매기살','갈비','곱창','꽃등심','대창','돈가스','등갈비','등뼈','등심','떡갈비','막창','목살','아롱사태','안심','양고기','양지다짐육','양지','양지머리','오돌뼈','오리고기','우둔살']
    basic_seasoning=['간장','다시마','다시다','다시육수','다시포','다시팩','국물용건멸치다시마팩','다시국물','멸치다시마육수','후추','설탕','소금','물','맛술','설탕','마늘','후추','소금']
    meet = pig + cow + chicken + etc_meet  ### 돼지고기 + 소고기 + 닭고기 + 기타 고기
    meet_and_seasoning = meet + basic_seasoning  ### 고기류 + 기본 조미료 => Word2Vec 기반 식재료 추천(코사인 유사도 계산) 시 input으로 고기가 나오면 meet_and_seasoning에 있는 것들은 제외하고 식재료 추천

    pig_set, cow_set, chicken_set, etc_meet_set, basic_seasoning_set, meet_set, meet_and_seasoning_set = set(pig), set(cow), set(chicken), set(etc_meet), set(basic_seasoning), set(meet), set(meet_and_seasoning)
    not_meet_or_seasoning_set = ingredient_dict.difference(meet_and_seasoning_set)  ### 전체 식재료 - (고기류+기본 조미료)
    not_seasoning_set = ingredient_dict.difference(basic_seasoning_set)  ### 전체 식재료 - 기본 조미료


    # (고기류, 고기류), (고기류, 기본 식재료), (기본 식재료, 기본 식재료) 제거
    a = []

    for c in range(len(temp_list)):
        if (temp_list[c][0] in meet_and_seasoning_set) & (temp_list[c][1] in meet_and_seasoning_set):
            a.append(c)
    for i in range(len(a)):
        a[i] = a[i] - i

    for i in a:
        temp_list.pop(i)


    # 조합에 기본 식재료 들어가는 경우 제거
    b = []

    for c in range(len(temp_list)):
        if (temp_list[c][0] in basic_seasoning_set) or (temp_list[c][1] in basic_seasoning_set):
            b.append(c)
    for i in range(len(b)):
        b[i] = b[i] - i


    # 이제 temp_list에서 pop해주면 됨
    for i in b:
        temp_list.pop(i)

    for combination in temp_list:
        combinations_dict[combination] = support(combination[0],combination[1], recipes)

    combinations_dict = sorted(combinations_dict.items(), key=lambda x: x[1], reverse=True)


    # support가 0인 것들은 제외
    remove_idxs = []
    for idx in range(len(combinations_dict)):
        if combinations_dict[idx][1] == 0:
            remove_idxs.append(idx)

    remove_idxs.reverse()

    for remove_idx in remove_idxs:
        combinations_dict.pop(remove_idx)
        
    
    # 튜플값이 동일한 것들은 제외
    remove_idxs = []
    for idx in range(len(combinations_dict)):
        if combinations_dict[idx][0][0] == combinations_dict[idx][0][1]:
            remove_idxs.append(idx)

    remove_idxs.reverse()

    for remove_idx in remove_idxs:
        combinations_dict.pop(remove_idx)


    # TOP N개만 필터링
    combinations_dict = combinations_dict[:topn]

    if len(combinations_dict) < topn:
        print(f'===== 식별된 유의미한 조합(Support > 0)은 {len(combinations_dict)}개만 존재합니다. =====')


    # topn개의 음식 조합 이중 리스트 형태로 저장하기
    topn_combinations = []
    for i in range(len(combinations_dict)):
        topn_combinations.append(list(combinations_dict[i][0]))

    return combinations_dict, topn_combinations



########### 메뉴 추천시스템 구현 ###########
def recommend_recipe_asr(recipe_dict, combinations_list, topn=10):


    # 초기화
    result = pd.DataFrame()
    result_temp = pd.DataFrame()
    recipe_names = list(recipe_dict.keys())


    # 조합 하나씩 불러온 후 데이터프레임에 추가
    for combination in combinations_list:
        ingredients = combination

        for idx, recipe_name in enumerate(recipe_names):

            recipe_ingredient = set(recipe_dict[recipe_name])
            intersections = list(set(recipe_dict[recipe_name]).intersection(ingredients))  # 교집합
            differences = list(set(recipe_dict[recipe_name]).difference(ingredients))  # 차집합 (특정 레시피를 위해 추가적으로 필요한 재료)

            if len(intersections) == len(ingredients):
                result_temp.loc[idx, '레시피'] = recipe_name
                result_temp.loc[idx, 'ASR 기반 추천 식재료 조합'] = ','.join(intersections)
                result_temp.loc[idx, '추가 필요 재료'] = ','.join(differences)

        result = pd.concat([result, result_temp], axis=0)


    # 중복제거 및 topn개만 불러오기
    result.drop_duplicates(inplace=True)
    final_result = result.reset_index(drop=True).iloc[:topn, :]
    
    if final_result.shape[0] < topn:
        print(f'===== 추천가능한 레시피의 최대 개수는 {final_result.shape[0]}개입니다. =====')

    return final_result