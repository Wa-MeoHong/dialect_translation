"""
	파일 : preprocessing2.py
	이름 : 신대홍
	날짜 : 23/08/02
	목적 : preprocessing1에서 전처리한 데이터의 개수를 맞춰주는 작업(다운 샘플링)
"""

import fire				# 이게 있어야 메인 함수를 따로만들어서, 파라미터를 집어넣고, 그걸 메인 코드를 실행 가능
import os
import re
import json
from typing import List, Union, Dict, Any
from datasets import load_dataset, load_from_disk, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def load_dialects_df(file_path):
	dataset = load_from_disk(file_path + "/dialect_data1") # 데이터셋 로드, 확인이 쉽게 먼저 DataFrame 형태로 다시 로드
	dataset2 = load_from_disk(file_path + "/dialect_data2") 
	
	df_train1 = (pd.DataFrame(dataset["train"])).drop(columns=['id','tgt_lang'], axis=1)
	df_valid1 = (pd.DataFrame(dataset["valid"])).drop(columns=['id','tgt_lang'], axis=1)
	df_train2 = pd.DataFrame(dataset2["train"])
	df_valid2 = pd.DataFrame(dataset2["valid"])
 
	df_train_con = pd.concat([df_train1, df_train2])
	df_valid_con = pd.concat([df_valid1, df_valid2])
 
	return df_train_con, df_valid_con

def data_fixed(train_set, valid_set):
	train_gw = train_set[(train_set['src_lang'] == 'gangwondo')]
	train_cc = train_set[(train_set['src_lang'] == 'chungchengdo')]
	trian_jn = train_set[(train_set['src_lang'] == 'jeolla')]
	train_gs = train_set[(train_set['src_lang'] == 'gyeongsangdo')]
 
	# 충청도 데이터셋 중, 12000개를 Valid_set으로 옮길 것 (Train_set의 0.055 정도였음)
	cc_train_split, cc_valid_split = train_test_split(train_cc, test_size=0.055, random_state=1231)
	
	valid_gw = valid_set[(valid_set['src_lang'] == 'gangwondo')]
	valid_cc = valid_set[(valid_set['src_lang'] == 'chungchengdo')]
	valid_jn = valid_set[(valid_set['src_lang'] == 'jeolla')]
	valid_gs = valid_set[(valid_set['src_lang'] == 'gyeongsangdo')]
 
	cc_valid_fixed = pd.concat([valid_cc, cc_valid_split])
 
	train_fix = pd.concat([train_gw, cc_train_split, trian_jn, train_gs])
	valid_fix = pd.concat([valid_gw, cc_valid_fixed, valid_jn, valid_gs])

def data_downsampling(dataset):
	"""
		중요 !
		이번에 사용되는 데이터셋은 전부 합쳐서 다음의 갯수를 가지고 있습니다.
			한국어 발화 데이터 + 중노년층 경상도 1인 발화 따라말하기 데이터셋
		강원도 dataset : Train (500209개), Valid(59574개)
		전라도 dataset : Train (291535개), Valid(57241개)
		충청도 dataset : Train (214000여 개), Valid(45482개)
  		경상도 dataset : Train (180498개), Valid(43806개)

		따라서, 다음과 같이 설정되었습니다. 꼭, 데이터셋이 달라진다면, 확인을 하고, 다른 비율을 적용해야합니다.
		
		가장 코퍼스 개수가 적었던 경상도 데이터셋을 기준으로 작성하였습니다.
	"""	
	df_gangwon = dataset[(dataset['src_lang'] == 'gangwondo')]
	df_chungcheng = dataset[(dataset['src_lang'] == 'chungchengdo')]
	df_jeolla = dataset[(dataset['src_lang'] == 'jeolla')]
	df_gyeongsang = dataset[(dataset['src_lang'] == 'gyeongsangdo')]
 
	# 데이터를 랜덤하게 경상도 코퍼스 개수 정도 만큼 샘플링
	df_gangwon_re = df_gangwon.sample(frac=round((len(df_gyeongsang)/len(df_gangwon)),2), random_state=1023)
	df_chungcheng_re = df_chungcheng.sample(frac=round((len(df_gyeongsang)/len(df_chungcheng)), 2), random_state=1023)
	df_jeolla_re = df_jeolla.sample(frac=round((len(df_gyeongsang)/len(df_jeolla)), 2), random_state=1023)

	# 데이터셋을 하나로 연결한 뒤 셔플로 섞어줌
	df_downsampled = (pd.concat([df_gangwon_re, df_chungcheng_re, df_jeolla_re, df_gyeongsang]))
	df_downsampled = df_downsampled.sample(frac=1).reset_index(drop=True)
 
	return df_downsampled
	
def main(
	file_path : str = "",
	dest_path : str = ""
):
    # 만약 주소들이 입력이 안되었으면 실행 X 
	assert ( 
        file_path or dest_path
    ), "데이터가 들어있는 파일의 상대 경로를 --file_path './경로' 에 맞게 입력해주세요.\n \
        저장할 디렉토리를 --dest_path './경로'에 맞게 입력해주세요."
	
    # 데이터셋 로드
	train_set, valid_set = load_dataset(file_path)
    
    # 중복 데이터 제거
	train_set.drop_duplicates(subset=['source'], inplace=True)
	valid_set.drop_duplicates(subset=['source'], inplace=True)
    # 결측치를 제거
	train_set.dropna(subset=['source'], inplace=True)
	valid_set.dropna(subset=['source'], inplace=True)
    
    # 충청도 데이터 개수 보강
	train_set_fix, valid_set_fix = data_fixed(train_set, valid_set)
	
	# 데이터 다운샘플링
	train_re = data_downsampling(train_set_fix)
	valid_re = data_downsampling(valid_set_fix)
	
	# Valid 데이터셋에서 valid, test를 나눔 (절반씩 나눠가짐)
	df_valid_div, df_test_div = train_test_split(valid_re, test_size=0.5, random_state=123)
	df_valid_div = df_valid_div.sample(frac=1).reset_index(drop=True)
	df_test_div = df_test_div.sample(frac=1).reset_index(drop=True)
 
	# 데이터셋 딕셔너리 형태로 변환
	train_dataset = Dataset.from_dict(train_re)
	valid_dataset = Dataset.from_dict(df_valid_div)
	test_dataset = Dataset.from_dict(df_test_div)
	
	# json 형태로 저장
	train_dataset.to_json(dest_path + '/train_clean.json', force_ascii=False)
	valid_dataset.to_json(dest_path + '/valid_clean.json', force_ascii=False)
	test_dataset.to_json(dest_path + '/test_clean.json', force_ascii=False)
	
if __name__ == "__main__":
    fire.Fire(main)