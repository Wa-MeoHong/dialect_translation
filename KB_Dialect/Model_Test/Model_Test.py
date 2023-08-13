import os.path as osp
import sys

import fire
import pandas as pd
import numpy as np
import torch
import transformers
from peft import PeftModel
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

# Alpaca-QLoRA util file 로드위한 경로
path = osp.dirname(osp.dirname(osp.realpath(__file__)))
sys.path.append(f"{path}/Alpaca-QLoRA")

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
from utils.smart_tokenizer import smart_tokenizer_and_embedding_resize

from params import *

from huggingface_hub import login
login(token="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")

def Test_model(
    load_8bit : bool = False,
    base_model : str = "",
    lora_weights : str = "",
    test_sets : str = "", 
    prompt_template : str = "custom_template",
    server_name : str = "0.0.0.0",
    use_landmark : bool = False,
    use_scaled_rope : bool = False,
    use_ntk_aware_scaled_rope : bool = False,
    max_new_tokens : int = 64
):
    assert base_model, "Please fill base_model ( --base_model '원본 모델(허깅페이스 주소)' )\n"
    assert lora_weights, "Please fill lora_weights ( --lora_weights 'lora 모델 디렉토리 주소' )\n"
    assert test_sets, "Please fill test_sets ( --test_sets '테스트 셋이 들어있는 json 주소' )\n"
    
    #base_model = 'EleutherAI/polyglot-ko-12.8b'
    #lora_weights = 'Meohong/Dialect-Polyglot-12.8b-QLoRA'
    #test_sets = '/content/drive/MyDrive/한국어_방언_데이터셋/data/dialect_cleaning/prompt/cutoff_set/방언번역_cut_test_set_prompt.json' 
    
    prompter = Prompter(prompt_template)                         # 프롬프트 로드
    tokenizer = AutoTokenizer.from_pretrained(base_model)     # 토크나이저 로드
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        #load_in_8bit=load_8bit,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
        )
    
    # PEFT 모델로 로드
    if lora_weights is not None:
        model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
        # device_map={'': 0}
    )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    # 모델이 입력 문장에 대해 번역을 실행하는 함수 
    def evaluate(
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=max_new_tokens,
        **kwargs,
    ):    
        instruction = "사투리가 포함된 문장이면 표준어로 변환해주시오."
        prompt = prompter.generate_prompt(instruction, input)       # 프롬프트에 맞게, input id, attention mask, position encoding 진행
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                **kwargs,
            )
        
        generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        output = output.replace("<|endoftext|>", "")
        result = prompter.get_response(output)
        print("predict complete : ", result)
        return result     
    
    # 데이터셋 
    test_set = pd.DataFrame(load_dataset("json", data_files={"test" : test_sets})["test"])
    test_set = test_set.sample(frac=0.2, random_state=1243)
    
    # 테스트용 입력 데이터와, 실제 예측되어야하는 데이터값을 분리
    test_input = test_set["input"].tolist()
    test_reg = test_set["output"].tolist()
    
    # 예측 문장 산출
    predicts = []
    for input in test_input:
        predicts.append(evaluate(input))
    
    # BLEU 점수 계산 (sacrebleu)
    metric = load_metric("sacrebleu")    
    
    # 각 문장에 관해 점수를 계산한다. 
    BLEU_scores = []
    for predict in predicts:
        BLEU_scores.append(metric.compute(predictions=[predict], references=[test_reg])['score'])

    BLEU_mean = np.mean(BLEU_scores)
    print(BLEU_mean)
    
if __name__ == "__main__":
    fire.Fire(Test_model)

