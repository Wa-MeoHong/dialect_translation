# dialect_translation <br>한국어 방언 -> 표준어 번역 AI 모델 만들기

## 1. 개요 <br>
KB 제 5회 Future Finance A.I. challenge 참여하면서, 주제로 상담 시, 사투리를 많이 쓰는 지방 사람들과 대화할 시, 사투리를 표준어로 변환하여, 좀 더 수월한 대화를 가능케 하기 위해 제안했음.<br>
(이후, 개요 부분은 아이디어의 구상 내용 뒷받침을 추가할 예정)

## 2. 활용 데이터셋
AI-Hub에서 제공하고 있는 '한국어 발화 데이터셋 (경상도, 강원도, 충청도, 전라도)' + '중,노년층 발화 데이터 (경상도) 중 1인 따라부르기 발화 셋'을 활용하였습니다. 발화 데이터셋 원본은 공유가 불가능하지만, Data_preprocessing을 통해, 사투리만 포함된 데이터만 뽑아내고, 
Data Downsampling을 통해 각 4개 도의 사투리 발화 Corpus 개수를 균일하게 맞췄다.<br>
원래 Downsampling을 거친 데이터셋의 총 량은 거의 80만개 Corpus를 가지고 있었으나,<br>
학습의 시간과 Colab runtime Error 때문에 부득이하게 데이터 개수를 1/8배하여 거의 10만개의 코퍼스를 가지고 진행하게 되었다. <br>
> ### 최종적으로 사용된 코퍼스 개수 
> #### Train : <br>
> * 강원도 : 23668 Corpus
> * 전라도 : 23512 Corpus
> * 충청도 : 23185 Corpus
> * 경상도 : 23432 Corpus
> 
> #### Valid / Test : <br>
> * 강원도 : 5049 / 5135 Corpus
> * 전라도 : 5015 / 5079 Corpus
> * 충청도 : 5227 / 5065 Corpus
> * 경상도 : 4935 / 4947 Corpus
<br>

## 3. 활용 모델 (Polyglot-ko-12.8b)<br> 
이번에 사용한 모델은 [**Polyglot-ko-12.8b**](https://huggingface.co/EleutherAI/polyglot-ko-12.8b) 모델을 사용하였다. 이 모델이 선택된 이유는 다음과 같습니다.<br>
<ol>
  1. GPT-NeoX 기반으로 한국어 데이터셋 (863G)으로 Pre-Trained 된 
</ol>
