"""
   파일 : preprocessing1.py
   날짜 : 23/07/30
   이름 : 신대홍
   참고파일 : KoBART-dialect prepare_data.py
   목적 : AI-Hub에서 공개중인 한국어 방언 발화 데이터(제주도 제외 4개도) 내
        사투리 발화만을 찾아내는 것.
"""
import os
import os.path as osp
import json
from json import JSONDecodeError
from glob import glob
from tqdm import tqdm
from typing import List, Union, Dict, Any
# 주의 : pip install datasets==2.7.1 해야함.
# datasets 2.8.0 부터 Batch 클래스 없어짐
from datasets import load_dataset, load_from_disk
from datasets.arrow_dataset import Batch

from param import *

# 데이터내 특수문자 및 줄바꿈문자를 대체 및 삭제
def clean(s):
    s = s.replace("\n", "")
    s = s.replace("\t", "")
    s = re.sub(", +}", ",}", s)
    s = re.sub(", +]", ",]", s)
    s = s.replace(",}", "}")
    s = s.replace(",]", "]")
    s = s.replace("'", "\"")
    s = s.replace(".,", ",")
    return s
     
# 사투리와 표준어 문장이 섞인 원본데이터셋에서 사투리 발화만 걸러내는 전처리 작업 함수들
def prepare_dialect_dataset(filenames: List[str]):
    results = []
    for filename in tqdm(filenames):                    # 모든 json 파일을 열기 때문에 for문을 사용했음
      with open(filename, encoding="utf-8-sig") as f:   # file을 읽음
          s = f.read()
      try:
        data = json.loads(s)
      except JSONDecodeError:
        data = json.loads(clean(s))

      # 메타 데이터나, 다른 모든 정보는 빼고, 발화/대화만 뽑음
      utterance = data["utterance"]
      for u in utterance:
      # 사투리가 들어간 문장일 때만, 필터링( 사투리 없는 문장은 걸러짐 )
        if u["standard_form"] != u["dialect_form"]:
          dialect_idx = []
          for e in u["eojeolList"]:       # 어절 안에서 사투리가 있다면, 인덱스를 추가.
            if e["isDialect"]:
              dialect_idx.append(e["id"])
          # 사투리가 든 문장을 샘플로 다시 만들고, result에 넣음.
          sample = {
                    "id": u["id"],
                    "do": filename.split("/")[1],
                    "standard": u["standard_form"],
                    "dialect": u["dialect_form"],
                    "dialect_idx": dialect_idx,
          }
          results.append(sample)

    return results
# 중요! 중노년층 방언 발화 중 1인 따라부르기 데이터셋만 가져온다.
def prepare_dialect_dataset2(filenames: List[str]):
    results = []
    for filename in tqdm(filenames):          # 모든 json 파일을 열기 때문에 for문을 사용했음
      with open(filename, encoding="utf-8-sig") as f:   # file을 읽음
        s = f.read()
        try:
          data = json.loads(s)
        except JSONDecodeError:
          data = json.loads(clean(s))

      # 따라부르기이기 때문에 모든 문장이 사투리가 포함되어있음
      utterance = data["transcription"]
      # 사투리가 든 문장을 샘플로 다시 만들고, result에 넣음.
      sample = {
                    "source": utterance["dialect"],
                    "target": utterance["standard"],
                    "src_lang": "gyeongsangdo"
      }
      results.append(sample)

    return results

# KB-Dialect 폴더 안에 저장해야 데이터 인식이 가능
def data_to_json1(inputs, dest_path):
    file_path = osp.dirname(osp.dirname(osp.realpath(__file__)))
    
    # train, valid 셋에 관해서 저장을 달리 하기 때문에 조건문에서 다르게 처리한다.
    if inputs == 'train':
        # 만약 이미 파일이 있다면 즉시 리턴으로 파일을 불러오지 않는다.
        if osp.isfile("./train_dataset1.json"):
            return 
        files = glob(file_path+"/'한국어발화1'/*/train/*.json")     # glob : 다음을 만족하는 파일 경로들을 리스트의 형태로 반환
        datas = prepare_dialect_dataset(files)        # 파일 경로 리스트를 통해 사투리 데이터를 전부 뽑아온다.
        # json 파일로 다시 저장한다. (사투리 발화만 존재하는 데이터셋)
        json.dump({"data" : datas}, open(dest_path + "/train_dataset1.json", "w"))
        return
        
    elif inputs == 'valid':
        # 만약 이미 파일이 있다면 즉시 리턴으로 파일을 불러오지 않는다.
        if osp.isfile("./valid_dataset1.json"):
            return 
        files = glob(file_path+"/'한국어발화1'/*/valid/*.json")   # glob : 다음을 만족하는 파일 경로들을 리스트의 형태로 반환
        datas = prepare_dialect_dataset(files)      # 파일 경로 리스트를 통해 사투리 데이터를 전부 뽑아온다.
        # json 파일로 다시 저장한다. (사투리 발화만 존재하는 데이터셋)
        json.dump({"data" : datas}, open(dest_path +"/train_dataset1.json", "w"))
        return   
def data_to_json2(inputs, dest_path):
    file_path = osp.dirname(osp.dirname(osp.realpath(__file__)))
    
    # train, valid 셋에 관해서 저장을 달리 하기 때문에 조건문에서 다르게 처리한다.
    if inputs == 'train':
        # 만약 이미 파일이 있다면 즉시 리턴으로 파일을 불러오지 않는다.
        if osp.isfile(dest_path + "/train_dataset2.json"):
            return 
        files = glob(file_path + "/'한국어발화2'/*/train/*.json")     # glob : 다음을 만족하는 파일 경로들을 리스트의 형태로 반환
        datas = prepare_dialect_dataset(files)        # 파일 경로 리스트를 통해 사투리 데이터를 전부 뽑아온다.
        # json 파일로 다시 저장한다. (사투리 발화만 존재하는 데이터셋)
        json.dump({"data" : datas}, open(dest_path +"/train_dataset2.json", "w"))
        return
        
    elif inputs == 'valid':
        # 만약 이미 파일이 있다면 즉시 리턴으로 파일을 불러오지 않는다.
        if osp.isfile(dest_path +"/valid_dataset2.json"):
            return 
        files = glob(file_path+"/'한국어발화2'/*/valid/*.json")   # glob : 다음을 만족하는 파일 경로들을 리스트의 형태로 반환
        datas = prepare_dialect_dataset(files)      # 파일 경로 리스트를 통해 사투리 데이터를 전부 뽑아온다.
        # json 파일로 다시 저장한다. (사투리 발화만 존재하는 데이터셋)
        json.dump({"data" : datas}, open(dest_path + "/train_dataset2.json", "w"))
        return

# 사투리 발화만 있는 데이터셋을 정규표현식으로 걸러준 후, 그 DatasetDict를 저장하는 함수
def preprocess1(examples: Batch) -> Union[Dict, Any] :
    # Example마다 전부 긁어오고, 데이터를 정규표현식에 맞춰서 전부 바꾼 후, 다시 new_example을 반환
    ids = examples["id"]
    dos = examples["do"]
    standard_texts = examples["standard"]
    dialect_texts = examples["dialect"]
    dialect_idxs = examples["dialect_idx"]

    new_examples = {"id": [], "do": [], "standard": [], "dialect": [], "dialect_idx": []}

    iterator = zip(ids, dos, standard_texts, dialect_texts, dialect_idxs)

    # 각 문장마다 전처리
    for _id, do, standard_text, dialect_text, dialect_idx in iterator:
        # 특수문자 삭제, 패턴 1~4 모두
        # remove (()), {\w+} patterns
        standard_text = re.sub(PATTERN1, "", standard_text).replace("  ", " ")
        dialect_text = re.sub(PATTERN1, "", dialect_text).replace("  ", " ")
        # remove \n, \t patterns
        standard_text = re.sub("[\t\n]", " ", standard_text).replace("  ", " ")
        dialect_text = re.sub("[\t\n]", " ", dialect_text).replace("  ", " ")
        # remove sample which has ((\w+)) patterns
        if PATTERN2.findall(standard_text) + PATTERN2.findall(dialect_text):
            continue
        # $\w+$ pattern mapping
        for k in PATTERN3.findall(standard_text):
            standard_text = standard_text.replace(k, PAT_MAP.get(k, "[OHTER]"))
        for k in PATTERN3.findall(dialect_text):
            dialect_text = dialect_text.replace(k, PAT_MAP.get(k, "[OHTER]"))
        # (\w+)/(\w+)
        standard_text = re.sub(PATTERN4, r"\2", standard_text)
        dialect_text = re.sub(PATTERN4, r"\1", dialect_text)

        new_examples["id"].append(_id)
        new_examples["do"].append(do)
        new_examples["standard"].append(standard_text)
        new_examples["dialect"].append(dialect_text)
        new_examples["dialect_idx"].append(dialect_idx)

    return new_examples
def preprocess1_transfer(examples: Batch) -> Union[Dict, Any]:
    # 기본 Example의 값을 전부 받아오고,
    ids = examples["id"]
    dos = examples["do"]
    standard_texts = examples["standard"]
    dialect_texts = examples["dialect"]

    # 새로운 Example 탬플릿 딕셔너리 생성
    new_examples = {"id": [], "source": [], "target": [], "src_lang": [], "tgt_lang": []}

    # 아까 분리시킨 기존 Example요소를 모두 zip한다.
    iterator = zip(ids, dos, standard_texts, dialect_texts)

    #반복문을 통해, source, target을 서로 바꾼 데이터 example을 만들고 extend로 이어준다.
    for _id, do, standard_text, dialect_text in iterator:
        new_examples["id"].extend([_id])
        new_examples["source"].extend([dialect_text])
        new_examples["target"].extend([standard_text])
        new_examples["src_lang"].extend([do])
        new_examples["tgt_lang"].extend(["standard"])
        
    return new_examples
def data_preprocess1(file_path):
    # 사투리 발화만 가져온 데이터셋을 DatasetDict형태로 로드해서, 
    data_files = {"train": file_path + "/train_dialect1.json", "valid": file_path + "/valid_dialect1.json"}
    dialect = load_dataset("json", data_files=data_files, field='data')
    
    dialect_process = dialect.map(function=preprocess1, batched=True, batch_size=1000)
    dialect_transfer = dialect_process.map(function=preprocess1_transfer, batched=True, batch_size=1000)
    
    dialect_transfer.save_to_disk(file_path + "/dialect_data1")
    
def preprocess2(examples: Batch) -> Union[Dict, Any] :
    standard_texts = examples["target"]
    dialect_texts = examples["source"]
    src_lang = examples["src_lang"]

    new_examples = {"source":[] ,"target": [], "src_lang": []}

    iterator = zip( standard_texts, dialect_texts, src_lang)

    # 각 문장마다 전처리
    for standard_text, dialect_text, src_lang in iterator:
       # 특수문자 삭제, 패턴 1~4 모두
        # remove (()), {\w+} patterns
        standard_text = re.sub(PATTERN1, "", standard_text).replace("  ", " ")
        dialect_text = re.sub(PATTERN1, "", dialect_text).replace("  ", " ")
        # remove \n, \t patterns
        standard_text = re.sub("[\t\n]", " ", standard_text).replace("  ", " ")
        dialect_text = re.sub("[\t\n]", " ", dialect_text).replace("  ", " ")
        # remove sample which has ((\w+)) patterns
        if PATTERN2.findall(standard_text) + PATTERN2.findall(dialect_text):
            continue
        # $\w+$ pattern mapping
        for k in PATTERN3.findall(standard_text):
            standard_text = standard_text.replace(k, PAT_MAP.get(k, "[OHTER]"))
        for k in PATTERN3.findall(dialect_text):
            dialect_text = dialect_text.replace(k, PAT_MAP.get(k, "[OHTER]"))
        # (\w+)/(\w+)
        standard_text = re.sub(PATTERN4, r"\2", standard_text)
        dialect_text = re.sub(PATTERN4, r"\1", dialect_text)

        new_examples["target"].append(standard_text)
        new_examples["source"].append(dialect_text)
        new_examples["src_lang"].append(src_lang)

    return new_examples
def data_preprocess2(file_path):
    # 사투리 발화만 가져온 데이터셋을 DatasetDict형태로 로드해서, 
    data_files = {"train": file_path + "/train_dialect2.json", "valid": file_path + "/valid_dialect2.json"}
    dialect = load_dataset("json", data_files=data_files, field='data')
    
    dialect_process = dialect.map(function=preprocess2, batched=True, batch_size=1000) 
    dialect_process.save_to_disk(file_path + "/dialect_data2")