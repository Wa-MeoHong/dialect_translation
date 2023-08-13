import sys
import struct
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import fire

def GGML_Transform(
    dir_model : str = "",
    dest_path : str = "",
):
    ftype = 1
    
    # 먼저 모델을 로드함. 이번엔 토크나이저도 필요함.
    # GGML은 하나에 모델, 토크나이저 정보들을 전부 포함함

    # 모델 config.json을 로드하여, 하이퍼파라미터를 로드한다.
    with open(f"{dir_model}/config.json","r", encoding="utf-8") as f:
        hparams = json.load(f)
        print(f"open successed! {dir_model}/config.json")
    print(hparams)
        
    # 모델, 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(dir_model)
    print("load successed! tokenizer")
    model = AutoModelForCausalLM.from_pretrained(dir_model, low_cpu_mem_usage=True)
    print("load successed! model")
    
    list_vars = model.state_dict()
    # 모델 파라미터의 형태 출력, 따로 필요없을 수도 있음.
    for name in list_vars.keys():
        print(name, list_vars[name].shape, list_vars[name].dtype)
    fout = open(f"{dest_path}/ggml-model-f16.bin")

    # GGML model에 파라미터 기입 (16진수)
    fout.write(struct.pack("i", 0x67676d6c))
    fout.write(struct.pack("i", hparams["vocab_size"]))
    fout.write(struct.pack("i", hparams["max_position_embeddings"]))
    fout.write(struct.pack("i", hparams["hidden_size"]))
    fout.write(struct.pack("i", hparams["num_attention_heads"]))
    fout.write(struct.pack("i", hparams["num_hidden_layers"]))
    fout.write(struct.pack("i", int(hparams["rotary_pct"]*(hparams["hidden_size"]//hparams["num_attention_heads"]))))
    fout.write(struct.pack("i", hparams["use_parallel_residual"] if "use_parallel_residual" in hparams else True))
    fout.write(struct.pack("i", ftype))
    
    # TODO: temporary hack to not deal with implementing the tokenizer
    for i in range(hparams["vocab_size"]):
        text = tokenizer.decode([i]).encode('utf-8')
        fout.write(struct.pack("i", len(text)))
        fout.write(text)
        
    for name in list_vars.keys():
        data = list_vars[name].squeeze().numpy()
        
        n_dims = len(data.shape)
        
        # ftype == 0 -> float32, ftype == 1 -> float16
        ftype_cur = 0
        if ftype != 0:
            if name [-7:] == ".weight" and n_dims == 2:
                print("\tConverting to float16")
                data = data.astype(np.float16)
                ftype_cur = 1
            else:
                print("\tConverting to float32")
                data = data.astype(np.float32)
                ftype_cur = 0
        else:
            if data.dtype != np.float32:
                print("\tConverting to float32")
                data = data.astype(np.float32)
                ftype_cur = 0
                
        # Header 설정
        name_str = name.encode('utf-8')
        fout.write(struct.pack("iii", n_dims, len(name_str), ftype_cur))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims-1-i]))
        fout.write(name_str)
        
        # data 내보내기
        data.tofile(fout)
        
    # 파일 닫기
    fout.close()
    print("Done. Output file: " + dest_path + "/ggml-model-f16.bin")
    print("")
    
if __name__ == "__main__":
    fire.Fire(GGML_Transform)