import torch
import os.path as osp
import sys
import fire
import pandas as pd
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline
from Speechs import *
from params import *
# Alpaca-QLoRA util file 로드위한 경로
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
from utils.smart_tokenizer import smart_tokenizer_and_embedding_resize
import gradio as gr

"""
# 통신을 위해 서버를 열어야 함.
from socket import *

HOST = str(gethostbyname(gethostname()))
HOSTNAME = gethostname()
print(HOST, ", ",HOSTNAME)     # 서버의 IP 주소 출력(클라이언트 접속용)
# 마이크로 입력한 문자열을 서버로 전송하기 위해, 소켓 통신을 이용
serverSock = socket(AF_INET, SOCK_STREAM)
serverSock.bind((HOSTNAME, 9998))        # 서버 바인딩. (ip, port)튜플

print("접속 대기중")
serverSock.listen(1)
connectionSock, addr = serverSock.accept()

# 클라이언트 소켓이 서버에 접속할 경우, 새로운 소켓과, ip주소를 받음.
print(str(addr),'에서 접속하였습니다!')
"""
"""
ip = str(input("서버의 IP주소를 입력하시오 : "))
clientSock = socket(AF_INET, SOCK_STREAM)
print("접속 요청")
clientSock.connect((ip, 9998))
print("접속 완료!")
"""

 # 혐오 발언 스코어 
hate_scores = 0.0

def main(
    prompt_template : str = "custom_template",
    base_model : str = "/content/drive/MyDrive/KB 공모전/polyglot-ko-12.8b",
    lora_weights : str = 'Meohong/Dialect-Polyglot-12.8b-QLoRA',
    max_new_tokens : int = 64
):
    
    prompter = Prompter(prompt_template)                         # 프롬프트 로드
    # path = osp.dirname(osp.dirname(osp.realpath(__file__)))
    # sys.path.append(f"{path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)     # 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        #load_in_8bit=load_8bit,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
        )# 모델 로드
    # PEFT 모델로 로드
    if lora_weights is not None:
        model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
        # device_map={'': 0}
    )
    
    # 혐오 발언 분류 모델 로드
    hate_pipe = pipeline("text-classification", model="jh0802/Korean-Hate-KCBERT-base")
    
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    # 혐오 발언 점수 체크 함수
    def hate_speechs(info):
        global hate_scores
        if info["label"] == "Hate": 
            hate_scores = hate_scores + info["score"]
            strings = "욕설/비하 발언이 탐지되었습니다."
            
            if hate_scores >= 3.0:
                strings = "욕설/비하 발언 지수가 3점이 넘어 통화를 종료합니다."
                hate_scores = 0.0
                
            return hate_scores, strings
        else:
            strings = "욕설/비하 발언 X"
            return hate_scores, strings

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
        # 공용 instruction을 사용하고 있기 때문에, 따로 이렇게 지정
        instruction = "사투리가 포함된 문장이면 표준어로 변환해주시오."
        prompt = prompter.generate_prompt(instruction, input)       # 프롬프트에 맞게, input id, attention mask, position encoding 진행
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        # 생성을 하기 위한 파라미터 설정 
        generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                **kwargs,
            )
        
        # 답변 생성 ( 사투리 번역 )
        generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )
        # 사투리 번역 후, 토큰화된 단어 문장을 들고옴
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)        # 토큰화(숫자화)된 문장 Decode
        output = output.replace("<|endoftext|>", "")        # 맨 마지막에 오는 EOS토큰 삭제
        result = prompter.get_response(output)              # 맨 마지막에 나오는 번역문만 골라냄
        
        # 혐오 발언 탐지를 위해 pipeline으로 넘김.
        # 결과는 딕셔너리가 담긴 리스트형태로 반환되어 요소만 들고옴
        hate_predict = hate_pipe(result)[0]
        scores, strings = hate_speechs(hate_predict) # 점수를 계산함.
        
        # yield로 넘겨야 gradio에서 출력 가능
        yield result, strings, str(scores)
    
    gr.Interface(
        fn=evaluate,
        
        # Input 하는 곳
        inputs=[
            gr.components.Textbox(lines=2, label="Input", placeholder="none")
        ],
        # 출력 하는 곳, 위에서부터 번역본, 탐지 결과, 혐오발언 스코어 
        outputs=[
            gr.inputs.Textbox(
                lines=2,
                label="Output",
            ),
            gr.inputs.Textbox(
                lines=2,
                label="탐지"  
            ),
            gr.inputs.Textbox(
                lines=1,
                label="Hate score"
            )
        ],
        title="🌲 KB-Dialect ",
        description="한국어 사투리 번역과, 누적 Hate Score를 보여줍니다.",  # noqa: E501
    ).queue().launch(server_name="0.0.0.0", share=True)
    
if __name__ == "__main__":
    fire.Fire(main)
