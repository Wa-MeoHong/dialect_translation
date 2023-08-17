
# all imports

from IPython.display import Javascript, display
from google.colab import output
from base64 import b64decode
from io import BytesIO
from pydub import AudioSegment

RECORD = """
const sleep  = time => new Promise(resolve => setTimeout(resolve, time))
const b2text = blob => new Promise(resolve => {
  const reader = new FileReader()
  reader.onloadend = e => resolve(e.srcElement.result)
  reader.readAsDataURL(blob)
})
var record = time => new Promise(async resolve => {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  recorder = new MediaRecorder(stream)
  chunks = []
  recorder.ondataavailable = e => chunks.push(e.data)
  recorder.start()
  await sleep(time)
  recorder.onstop = async ()=>{
    blob = new Blob(chunks)
    text = await b2text(blob)
    resolve(text)
  }
  recorder.stop()
})
"""
import gradio as gr
import numpy as np
import speech_recognition as sr

# Recognizer 객체 생성 (음성 인식 객체)
Voice_record = sr.Recognizer()


def reverse_audio(audio):
    sound = Voice_record.record(audio)
    s
    text = Voice_record.recognize_google(sound, language='ko') 
    # 마이크로 녹음된 소리를 google API를 사용해 STT처리 (하루 50번) 한국어
    print("Text STT successed :", text) #출력
    return (sound, sound)

iface = gr.Interface(reverse_audio, "microphone", "audio")

iface.launch(debug=True)
