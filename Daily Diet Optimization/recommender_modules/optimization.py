###################################### Load packages ######################################
import cvxpy
import warnings
import ast

import pandas as pd
import numpy as np

warnings.filterwarnings(action='ignore')


###################################### optimization1 utils ######################################
def get_nutrient_coefficient_lowerbound(data, column):
    
    return np.array(list(-data[column]) + [0] * 6)
    
    
def get_nutrient_coefficient_upperbound(data, column, M = 1000000):
    
    arr = list(data[column]) + [0] * 6
    
    if column == '칼로리':
        arr[-6] = -M
        
    elif column == '탄수화물':
        arr[-5] = -M
        
    elif column == '단백질':
        arr[-4] = -M
        
    elif column == '나트륨':
        arr[-3] = -M
    
    elif column == '칼슘':
        arr[-2] = -M
        
    elif column == '비타민c':
        arr[-1] = -M
    
    return np.array(arr)
    

def get_food_allocation_coefficient(data, column):
    
    arr = [0] * (data.shape[0] + 6)
    for idx in data[data['카테고리'] == column].index:
        arr[idx] = 1
        
    return np.array(arr)



###################################### optimization2 utils ######################################
def constraint1(dataframe, idx, decision_variables):
    
    '''
    제약식1
     - 밥식은 서브디쉬가 3개이상 할당
     - 국밥/면/죽식은 서브디쉬가 2개이상 할당
     - 빵식은 서브디쉬가 4개만 할당
    '''
    
    arr = np.zeros(dataframe.shape[0] * dataframe.shape[1])
    
    if idx == 0:
        arr[0:int((arr.shape[0])/3)] += 1
    elif idx == 1:
        arr[int((arr.shape[0])/3):int((arr.shape[0])/3)*2] += 1
    elif idx == 2:
        arr[int((arr.shape[0])/3)*2:] += 1
    
    
    if '_밥' in dataframe.iloc[idx].name:
        return arr * decision_variables >= 3
    
    elif '_국밥/면/죽' in dataframe.iloc[idx].name:
        return arr * decision_variables >= 2
    
    elif '_빵' in dataframe.iloc[idx].name:
        return arr * decision_variables == 4


def constraint2(dataframe, constant, decision_variables):
    
    '''
    제약식2: 각 서브디쉬는 1개의 메인디쉬에만 할당
    '''

    arr = np.zeros(dataframe.shape[0] * dataframe.shape[1])
    
    arr[constant] = 1
    arr[int(arr.shape[0]/3) + constant] = 1
    arr[int(arr.shape[0]/3)*2 + constant] = 1
    
    return arr * decision_variables == 1


def find_servefoodtype_index(dataframe, element):
    
    return list(pd.Series(dataframe.columns)[pd.Series(dataframe.columns).str.contains(element)].index)


def find_mainfoodtype_index(dataframe, element):
    
    return list(pd.Series(dataframe.index)[pd.Series(dataframe.index).str.contains(element)].index)


def find_specific_index(dataframe, maindish_lists, servedish_lists):
    
    results = []
    
    for i in maindish_lists:

        tmp_list = []
        for j in servedish_lists:
            tmp_list.append(i*dataframe.shape[1] + j)

        results.append(tmp_list)
        
    return results


def constraint3(dataframe, lists, equal_type, decision_variables):
    
    '''
    1. <국>은 <밥>류와 관련된 식단에서 1개만 할당된다.
    2. <반찬>은 <밥>류와 관련된 식단에서 1개이상 할당된다.
    3. <반찬>은 <국밥/면/죽>류와 관련된 식단에서 1개이상 할당된다.
    4. <샐러드>는 <빵>류와 관련된 식단에서 1개만 할당된다.
    5. <스프>는 <빵>류와 관련된 식단에서 1개만 할당된다.
    6. <후식>은 <밥>류와 관련된 식단에서 1개만 할당된다.
    7. <후식>은 <국밥/면/죽>류와 관련된 식단에서 1개만 할당된다.
    8. <후식>은 <빵>류와 관련된 식단에서 2개만 할당된다.
    '''
    
    arr = np.zeros(dataframe.shape[0] * dataframe.shape[1])
    
    arr[lists] = 1
    
    if equal_type == '>=1':
        return arr * decision_variables >= 1
    elif equal_type == '==1':
        return arr * decision_variables == 1
    elif equal_type == '==2':
        return arr * decision_variables == 2


###################################### optimize1 algorithm ######################################
def optimize1(dataframe, b1):
    
    ########################### 결정변수 ##########################

    '''
    1. 전체 레시피 선정에 대한 이진변수 (각 요리별)
    2. 초과여부에 대한 이진변수 (각 영양소별)
    '''

    num_variables = dataframe.shape[0] + 6
    variables =  cvxpy.Variable(num_variables, boolean = True)

    ########################### 목적함수 ##########################
    objective_function_coefficient = np.array([0] * (dataframe.shape[0]) + [1] * 6)
    objective_function = objective_function_coefficient * variables

    ########################### 제약식 ##########################
    weight_constraints = []

    ##### 1. unequal 제약식 (영양소 하한~상한 제약식 + 반찬개수는 1개이상)
    A1 = [get_nutrient_coefficient_lowerbound(dataframe, '칼로리'), 
          get_nutrient_coefficient_lowerbound(dataframe, '탄수화물'), 
          get_nutrient_coefficient_lowerbound(dataframe, '단백질'),
          get_nutrient_coefficient_lowerbound(dataframe, '나트륨'), 
          get_nutrient_coefficient_lowerbound(dataframe, '칼슘'), 
          get_nutrient_coefficient_lowerbound(dataframe, '비타민c'),
          get_nutrient_coefficient_upperbound(dataframe, '칼로리', M=10000),
          get_nutrient_coefficient_upperbound(dataframe, '탄수화물', M=10000),
          get_nutrient_coefficient_upperbound(dataframe, '단백질', M=10000),
          get_nutrient_coefficient_upperbound(dataframe, '나트륨', M=10000),
          get_nutrient_coefficient_upperbound(dataframe, '칼슘', M=10000),
          get_nutrient_coefficient_upperbound(dataframe, '비타민c', M=10000),
          get_food_allocation_coefficient(dataframe, '밥') + get_food_allocation_coefficient(dataframe, '국밥/면/죽') - get_food_allocation_coefficient(dataframe, '반찬')
        ]
    
    b1.append(0)

    for idx in range(len(A1)):
        weight_constraints.append(A1[idx] * variables <= b1[idx])

    ##### 2. equal 제약식
    A2 = [get_food_allocation_coefficient(dataframe, '밥') + get_food_allocation_coefficient(dataframe, '국밥/면/죽') + get_food_allocation_coefficient(dataframe, '빵'), 
          get_food_allocation_coefficient(dataframe, '국') - get_food_allocation_coefficient(dataframe, '밥'),
          get_food_allocation_coefficient(dataframe, '샐러드') - get_food_allocation_coefficient(dataframe, '빵'),
          get_food_allocation_coefficient(dataframe, '스프') - get_food_allocation_coefficient(dataframe, '빵'),
          get_food_allocation_coefficient(dataframe, '후식') - get_food_allocation_coefficient(dataframe, '밥') - get_food_allocation_coefficient(dataframe, '국밥/면/죽') - get_food_allocation_coefficient(dataframe, '빵') * 2        
           ]

    b2 = [3, 
          0, 
          0, 
          0, 
          0]

    for idx in range(len(A2)):
        weight_constraints.append(A2[idx] * variables == b2[idx])

    ########################### 최적화 ##########################
    solver = cvxpy.Problem(cvxpy.Minimize(objective_function), weight_constraints)
    solver.solve(solver=cvxpy.GLPK_MI)
    
    
    
    ### 1차 최적화 결과 정리
    tmp = dataframe.copy()
    optimal_key = list(solver.solution.primal_vars.keys())[0]
    tmp['선정여부'] = list(map(lambda k: int(k), np.round(solver.solution.primal_vars[optimal_key])))[:-6]
    tmp2 = tmp[tmp['선정여부'] == 1].reset_index(drop=True)



    ### tag_vector 도출
    tag_names = ['면역력 강화', '간식/야식', '술안주', '해장요리', '손님 접대 요리', '나들이 요리', '파티/명절요리', '실생활요리', '한식 요리', '중식 요리', '일식 요리', '동남아/인도 요리', '멕시칸 요리', '양식 요리', '퓨전요리', '이국적인맛']
    tmp3 = tmp2[['레시피', '카테고리', '상황태그', '국가태그']]

    for idx in range(tmp3.shape[0]):

        tags = []

        try:
            tags.extend(tmp3.loc[idx, '상황태그'].split('|'))
        except:
            pass

        try:
            tags.extend(tmp3.loc[idx, '국가태그'].split('|'))
        except:
            pass

        tag_vector = [0] * 16

        for tag in tags:
            tag_index = tag_names.index(tag)
            tag_vector[tag_index] = 1

        tmp3.loc[idx, 'tag_vector'] = f'{tag_vector}'

    tmp4 = tmp3[['레시피', '카테고리', 'tag_vector']]



    ### 뽑힌 레시피간 조화로움 행렬 도출

    result = []
    for i in range(tmp4.shape[0]):

        semi_result = []
        for j in range(tmp4.shape[0]):

            value = np.dot(ast.literal_eval(tmp4.loc[i, 'tag_vector']), ast.literal_eval(tmp4.loc[j, 'tag_vector']))
            semi_result.append(value)

        result.append(semi_result)

    tmp5 = pd.DataFrame(result, index=tmp4['레시피'], columns=tmp4['레시피'])
    tmp5['카테고리'] = list(tmp2['카테고리'])


    ### 필요한 조화로움 행렬만 도출
    main_dishes = list(tmp2.loc[tmp2['카테고리'].isin(['밥', '국밥/면/죽', '빵']), '레시피'])
    main_dishes_index = list(tmp2.loc[tmp2['카테고리'].isin(['밥', '국밥/면/죽', '빵']), '레시피'].index)
    serve_dishes = list((tmp2['레시피']).drop(main_dishes_index)) + ['카테고리']
    tmp6 = tmp5.loc[main_dishes, serve_dishes]



    ### 변수명에 카테고리명 추가 (식별을 용이하게 하기 위함)
    column_change_dict = {f'{i}':f'{i}_{j}' for i, j in zip(tmp4['레시피'], tmp4['카테고리'])}
    tmp7 = tmp6.iloc[:, :-1]
    tmp7.columns = list(map(lambda x: column_change_dict[x], tmp6.columns[:-1]))
    tmp7.index = list(map(lambda x: column_change_dict[x], tmp6.index))
    
    return solver, optimal_key, tmp2, tmp7


###################################### optimize2 algorithm ######################################
def optimize2(dataframe):
    
    ########################### 결정변수 ##########################

    '''
    각 메인디쉬에 할당여부
    '''

    num_variables = dataframe.shape[0] * dataframe.shape[1]
    variables =  cvxpy.Variable(num_variables, boolean = True)

    ########################### 목적함수 ##########################
    objective_function_coefficient = np.array(dataframe).reshape(-1)
    objective_function = objective_function_coefficient * variables

    ########################### 제약식 ##########################
    weight_constraints = []

    ##### 1. 메인디쉬에 따라 할당되는 서브디쉬의 개수에 관련된 제약식 (행의 합에 관련된 제약식)
    for i in range(3):
        weight_constraints.append(constraint1(dataframe, i, variables))


    ##### 2. 각 서브디쉬는 1개의 식단에만 할당된다 (열의 합은 1)
    for i in range(dataframe.shape[1]):
        weight_constraints.append(constraint2(dataframe, i, variables))


    ##### 3. 각 서브디쉬의 할당과 관련된 제약식
    bab_indexes = find_mainfoodtype_index(dataframe, '_밥')
    gookbab_indexes = find_mainfoodtype_index(dataframe, '_국밥/면/죽')
    bread_indexes = find_mainfoodtype_index(dataframe, '_빵')
    gook_indexes = find_servefoodtype_index(dataframe, '_국')
    banchan_indexes = find_servefoodtype_index(dataframe, '_반찬')
    salad_indexes = find_servefoodtype_index(dataframe, '_샐러드')
    soup_indexes = find_servefoodtype_index(dataframe, '_스프')
    dessert_indexes = find_servefoodtype_index(dataframe, '_후식')

    ####### 3.1. 국
    gook_constraints = find_specific_index(dataframe, bab_indexes, gook_indexes)
    for gook_constraint in gook_constraints:
        weight_constraints.append(constraint3(dataframe, gook_constraint, '==1', variables))

    ####### 3.2. 반찬
    banchan_constraints = find_specific_index(dataframe, bab_indexes + gookbab_indexes, banchan_indexes)
    for banchan_constraint in banchan_constraints:
        weight_constraints.append(constraint3(dataframe, banchan_constraint, '>=1', variables))

    ####### 3.3. 샐러드
    salad_constraints = find_specific_index(dataframe, bread_indexes, salad_indexes)
    for salad_constraint in salad_constraints:
        weight_constraints.append(constraint3(dataframe, salad_constraint, '==1', variables))

    ####### 3.4. 스프
    soup_constraints = find_specific_index(dataframe, bread_indexes, soup_indexes)
    for soup_constraint in soup_constraints:
        weight_constraints.append(constraint3(dataframe, soup_constraint, '==1', variables))

    ####### 3.5. 디저트
    dessert_constraints = find_specific_index(dataframe, bab_indexes + gookbab_indexes, dessert_indexes)
    for dessert_constraint in dessert_constraints:
        weight_constraints.append(constraint3(dataframe, dessert_constraint, '==1', variables))

    dessert_constraints = find_specific_index(dataframe, bread_indexes, dessert_indexes)
    for dessert_constraint in dessert_constraints:
        weight_constraints.append(constraint3(dataframe, dessert_constraint, '==2', variables))

    ########################### 최적화 ##########################
    solver = cvxpy.Problem(cvxpy.Maximize(objective_function), weight_constraints)
    solver.solve(solver=cvxpy.GLPK_MI)
    optimal_key = list(solver.solution.primal_vars.keys())[0]
    
    return solver, optimal_key


###################################### Analysis Result ######################################
def analysis_result(harmony_matrix, all_df, solver_first, solver_second, key1, key2, b1, user_ingredient, ocr_ingredient):
    
    # 최적화 결과분석
    final_result = pd.DataFrame(solver_second.solution.primal_vars[key2].reshape(3, -1), index = harmony_matrix.index, columns = harmony_matrix.columns)
    final_result_t = final_result.transpose()

    maindish1 = final_result_t.columns[0]
    maindish2 = final_result_t.columns[1]
    maindish3 = final_result_t.columns[2]

    serve_dish1 = list(final_result_t.loc[final_result_t[maindish1] == 1, maindish1].index)
    serve_dish2 = list(final_result_t.loc[final_result_t[maindish2] == 1, maindish2].index)
    serve_dish3 = list(final_result_t.loc[final_result_t[maindish3] == 1, maindish3].index)

    meal1 = list(map(lambda x: x[:x.index('_')], [maindish1] + serve_dish1))
    meal2 = list(map(lambda x: x[:x.index('_')], [maindish2] + serve_dish2))
    meal3 = list(map(lambda x: x[:x.index('_')], [maindish3] + serve_dish3))
    
    
    
    # 칼로리 순으로 조식/중식/석식 결정
    nutrient_df = all_df[['레시피', '칼로리', '탄수화물', '단백질', '나트륨', '칼슘', '비타민c']]
    meal_selection = pd.DataFrame([
        list(nutrient_df.loc[nutrient_df['레시피'].isin(meal1), '칼로리':].sum().values) + [meal1],
        list(nutrient_df.loc[nutrient_df['레시피'].isin(meal2), '칼로리':].sum().values) + [meal2],
        list(nutrient_df.loc[nutrient_df['레시피'].isin(meal3), '칼로리':].sum().values) + [meal3]
    ], columns = ['칼로리(kcal)', '탄수화물(g)', '단백질(g)', '나트륨(mg)', '칼슘(mg)', '비타민c(mg)', '식단'])

    meal_selection = meal_selection.sort_values(by='칼로리(kcal)').reset_index(drop=True)
    meal_selection = meal_selection.append([meal_selection.sum()])
    meal_selection = round(meal_selection, 2)
    meal_selection.index = ['조식','중식','석식', '총합']

    

    # url 정보를 담은 데이터프레임 생성
    url_df = all_df[['레시피', 'url']]
    
    

    # 레시피별 영양소를 담은 데이터프레임 생성
    individual_foodinfo = round(all_df[['레시피', '칼로리', '탄수화물', '단백질', '나트륨', '칼슘', '비타민c']], 2)
    individual_foodinfo.loc[individual_foodinfo['레시피'].isin(meal_selection.loc['조식', '식단']), '식사종류'] = '조식'
    individual_foodinfo.loc[individual_foodinfo['레시피'].isin(meal_selection.loc['중식', '식단']), '식사종류'] = '중식'
    individual_foodinfo.loc[individual_foodinfo['레시피'].isin(meal_selection.loc['석식', '식단']), '식사종류'] = '석식'
    individual_foodinfo2 = individual_foodinfo.groupby(['식사종류', '레시피']).sum()
       
    order = []
    for meal_type in ['조식', '중식', '석식']:
        meal_type_list = list(individual_foodinfo.loc[individual_foodinfo['식사종류'] == meal_type, '레시피'])

        if set(meal_type_list) == set(meal1):
            for recipe in meal1:
                order.append((meal_type, recipe))

        elif set(meal_type_list) == set(meal2):
            for recipe in meal2:
                order.append((meal_type, recipe))

        elif set(meal_type_list) == set(meal3):
            for recipe in meal3:
                order.append((meal_type, recipe))
            
    individual_foodinfo3 = individual_foodinfo2.loc[order]
    individual_foodinfo3.columns = ['칼로리(kcal)', '탄수화물(g)', '단백질(g)', '나트륨(mg)', '칼슘(mg)', '비타민c(mg)']

    
    
    # 결과 출력
    print('=========================== 식재료들 정보 ==========================')
    print(f"\n(1) 사용자가 입력한 재료")
    print(f" → {', '.join(user_ingredient)}")
    print(f"\n(2) 전단지에서 OCR을 통해 식별된 재료")
    print(f" → {', '.join(ocr_ingredient)}")


    print('\n\n\n=========================== 1일 식단표 ===========================\n')
    print(f"조식: {', '.join(meal_selection.loc['조식', '식단'])}")
    print(f"중식: {', '.join(meal_selection.loc['중식', '식단'])}")
    print(f"석식: {', '.join(meal_selection.loc['석식', '식단'])}")

    
    print('\n\n\n=========================== 레시피 확인하기 ===========================')
    print('\n(1) 조식')
    for recipe, url in zip(meal_selection.loc['조식', '식단'], url_df.loc[url_df['레시피'].isin(meal_selection.loc['조식', '식단']), 'url']):
        print(f' - {recipe}: {url}')
    print('\n(2) 중식')
    for recipe, url in zip(meal_selection.loc['중식', '식단'], url_df.loc[url_df['레시피'].isin(meal_selection.loc['중식', '식단']), 'url']):
        print(f' - {recipe}: {url}')
    print('\n(3) 석식')
    for recipe, url in zip(meal_selection.loc['석식', '식단'], url_df.loc[url_df['레시피'].isin(meal_selection.loc['석식', '식단']), 'url']):
        print(f' - {recipe}: {url}')
    

    print('\n\n\n=========================== 구매가 필요한 재료들 분석 ===========================')
    ingredient_list = []
    all_dishes = all_df[['레시피', '변환된식재료']]
    for ingredient in all_dishes['변환된식재료'].tolist():
        tmp_list = ingredient.split(',')
        ingredient_list.extend(tmp_list)

    items_buy_from_leaflet = list(set(ingredient_list).intersection(set(ocr_ingredient)))
    items_need_to_buy_from_other = list(set(ingredient_list) - set(user_ingredient) - set(ocr_ingredient))

    print(f"\n(1) 전단지에서 구매가 필요한 재료")
    print(f" → {', '.join(items_buy_from_leaflet)}")
    print(f"\n(2) 그 외 추가구매가 필요한 재료")
    print(f" → {', '.join(items_need_to_buy_from_other)}")

    
    print('\n\n\n=========================== 레시피별 영양성분 정보 ===========================')
    display(individual_foodinfo3)


    print('\n\n\n=========================== 하루 권장섭취량 기준 1일 식단표 분석 ===========================\n')
    for name, unit, idx, boolean in zip(['칼로리', '탄수화물', '단백질', '나트륨', '칼슘', '비타민c'], ['kcal', 'g', 'g', 'mg', 'mg', 'mg'], [0, 1, 2, 3, 4, 5], np.array(np.round(solver_first.solution.primal_vars[key1][-6:]), dtype='int')):
        if boolean == 1:
            print(f"{idx+1}. {name} : {-b1[idx]}{unit} <= x <= {b1[idx+6]}{unit} ::::::::::::: 최적화 결과: {round(all_df[name].sum(), 2)} (초과)")
        else:
            print(f"{idx+1}. {name} : {-b1[idx]}{unit} <= x <= {b1[idx+6]}{unit} ::::::::::::: 최적화 결과: {round(all_df[name].sum(), 2)}")


    print('\n\n\n=========================== 식단별 영양성분 정보 ===========================')
    display(meal_selection.loc[:,'칼로리(kcal)':'비타민c(mg)'])
    
    
    
###################################### 최적화가 안되었을 때 ######################################
def exception_recipe_name_recommend(data, title): # 최적화가 안될 때 레시피명에 대한 log
    
    recipe_names = data['레시피'].tolist()

    if len(recipe_names) == 0:
        print(f"{title}: 요리가 가능한 레시피가 없습니다.")
    else:
        print(f"{title}: {', '.join(recipe_names)}")

def exception_url_recommend(data, title): # 최적화가 안될 때 url에 대한 log

    if data['레시피'].shape[0] == 0:
        print(f'\n{title} (요리가 가능한 레시피가 없습니다.)')

    else:
        print(f'\n{title}')
        for recipe, url, score in zip(data['레시피'], data['url'], data['해먹지수']):
            print(f" - {recipe} (해먹지수: {score}): {url}")

def exception_log(data, user_ingredient, ocr_ingredient):

    # 사용자로부터 보고자 하는 레시피의 개수 입력받기
    print('▲경고▲ 식재료의 부족으로 추천된 레시피만으로는 1일식단 구성이 불가능합니다!')
    print(' → 각 카테고리별로 요리 가능한 레시피를 해먹지수가 높은 순서로 제안합니다.')
    topn = int(input(' → 보고싶은 레시피의 최대 개수를 입력하세요: '))



    # 개수만큼 카테고리별로 커트 (해먹남녀 내림차순 기준)
    bab = data[data['카테고리'] == '밥'].sort_values(by='해먹지수', ascending=False).reset_index(drop=True)[:topn]
    gookbab = data[data['카테고리'] == '국밥/면/죽'].sort_values(by='해먹지수', ascending=False).reset_index(drop=True)[:topn]
    banchan = data[data['카테고리'] == '반찬'].sort_values(by='해먹지수', ascending=False).reset_index(drop=True)[:topn]
    bread = data[data['카테고리'] == '빵'].sort_values(by='해먹지수', ascending=False).reset_index(drop=True)[:topn]
    soup = data[data['카테고리'] == '스프'].sort_values(by='해먹지수', ascending=False).reset_index(drop=True)[:topn]
    salad = data[data['카테고리'] == '샐러드'].sort_values(by='해먹지수', ascending=False).reset_index(drop=True)[:topn]
    dessert = data[data['카테고리'] == '후식'].sort_values(by='해먹지수', ascending=False).reset_index(drop=True)[:topn]
    all_dishes = pd.concat([bab, gookbab, banchan, bread, soup, salad, dessert]).reset_index()



    # 식재료 정보 log
    print('\n\n\n=========================== 식재료들 정보 ==========================')
    print(f"\n(1) 사용자가 입력한 재료")
    print(f" → {', '.join(user_ingredient)}")
    print(f"\n(2) 전단지에서 OCR을 통해 식별된 재료")
    print(f" → {', '.join(ocr_ingredient)}")



    # 카테고리 정보 log
    print('\n\n\n=========================== 카테고리별 레시피 정보 ==========================\n')
    exception_recipe_name_recommend(bab, '(1) 밥')
    exception_recipe_name_recommend(gookbab, '(2) 국밥/면/죽')
    exception_recipe_name_recommend(banchan, '(3) 반찬')
    exception_recipe_name_recommend(bread, '(4) 빵')
    exception_recipe_name_recommend(soup, '(5) 샐러드')
    exception_recipe_name_recommend(salad, '(6) 스프')
    exception_recipe_name_recommend(dessert, '(7) 후식')



    # 레시피의 url 체크
    print('\n\n\n=========================== 레시피 확인하기 ===========================')
    exception_url_recommend(bab, '(1) 밥')
    exception_url_recommend(gookbab, '(2) 국밥/면/죽')
    exception_url_recommend(banchan, '(3) 반찬')
    exception_url_recommend(bread, '(4) 빵')
    exception_url_recommend(soup, '(5) 스프')
    exception_url_recommend(salad, '(6) 샐러드')
    exception_url_recommend(dessert, '(7) 후식')



    # 전단지에서 구매가 필요하거나 추가적으로 구매가 필요한 재료들 분석
    print('\n\n\n=========================== 구매가 필요한 재료들 분석 ===========================')
    ingredient_list = []
    for ingredient in all_dishes['변환된식재료'].tolist():
        tmp_list = ingredient.split(',')
        ingredient_list.extend(tmp_list)

    items_buy_from_leaflet = list(set(ingredient_list).intersection(set(ocr_ingredient)))
    items_need_to_buy_from_other = list(set(ingredient_list) - set(user_ingredient) - set(ocr_ingredient))

    print(f"\n(1) 전단지에서 구매가 필요한 재료")
    print(f" → {', '.join(items_buy_from_leaflet)}")
    print(f"\n(2) 그 외 추가구매가 필요한 재료")
    print(f" → {', '.join(items_need_to_buy_from_other)}")



    # 레시피별 영양성분 출력
    print('\n\n\n=========================== 레시피별 영양성분 정보 ===========================')
    all_dishes2 = all_dishes[['레시피', '카테고리', 'index', '칼로리', '탄수화물', '단백질', '나트륨', '칼슘', '비타민c']]
    all_dishes2.columns = ['레시피', '카테고리', ' ', '칼로리(kcal)', '탄수화물(g)', '단백질(g)', '나트륨(mg)', '칼슘(mg)', '비타민c(mg)']
    all_dishes2[' '] = all_dishes2[' '] + 1
    all_dishes2 = all_dishes2.set_index(['카테고리', ' '])
    display(all_dishes2)