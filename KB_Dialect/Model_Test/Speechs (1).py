"""
    Google API를 이용해, 마이크로 들려오는 목소리를 STT하여, 문자열로 바꾼다.
"""
import anvil.server

anvil.server.connect("server_5BJ2YOXG22HSXIMPKWKGKZS7-S34WPROM6CNKWZT5")

import speech_recognition as sr
# Recognizer 객체 생성 (음성 인식 객체)
Voice_record = sr.Recognizer()
    
@anvil.server.callable
def observe_voice():
    with sr.Microphone() as source:
        print("\t...Listening...")
        #audio = Voice_record.listen(source)
        text = input("->")
    try:
        #text = Voice_record.recognize_google(audio, language='ko') 
        # 마이크로 녹음된 소리를 google API를 사용해 STT처리 (하루 50번) 한국어
        print("Text STT successed :", text) #출력
        
        return text
    
    except sr.UnknownValueError:
        print("인식 실패") #음성인식 실패
        return 
        
    except sr.RequestError as e:
        print("요청 실패 : {0}".format(e)) #API key 오류, 네트워크 오류 등
        return 

anvil.server.wait_forever()
# if __name__ == "__main__":
#     observe_voice()