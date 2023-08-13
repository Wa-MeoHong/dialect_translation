"""
    파일 : MergeModel.py
    이름 : 신대홍
    날짜 : 23/08/05
    목적 : 제작된 LoRA 모듈을 붙여서, GGML로 변환하기 위해, 모델 병합을 하는 것 
"""

import fire
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

def transform(
    source_path : str = "",
    lora_path : str = "",
    dest_path : str = ""
):
    assert source_path, "Please fill source_path ( --source_path '원본 모델(허깅페이스 주소)' )\n"
    assert lora_path, "Please fill lora_path ( --lora_path 'lora 모델 디렉토리 주소' )\n"
    assert dest_path, "Please fill dest_path ( --dest_path '병합할 모델을 저장할 주소' )\n"
    
    # 원본 모델 로드
    base_model = AutoModelForCausalLM(
        source_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},         # device_map은 현재 CPU모델로 로드하기 위해 CPU로 설정
        trust_remote_code=True,
    )
    
    # LoRA 모델 로드 
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )
    
    # 원본과 Merge함, training 되는 경우를 막기 위해, weight들을 lock시킨다.
    lora_model = lora_model.merge_and_unload()
    lora_model.train(False)
    
    # 딕셔너리를 약간 수정하여, 병합한 모델을 그대로 불러올 수 있게 함.
    lora_model_sd = lora_model.state_dict()
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model_sd.items()
        if "lora" not in k
    }
    
    # 모델을 저장함. (원본모델을 기준으로 저장한다. 또한 1G만큼 잘라서, 저장한다.)
    base_model.save_pretrained(
        dest_path, state_dict=deloreanized_sd, max_shard_size="1024MB"
    )
    
if __name__ == "__main__":
    fire.Fire(transform)