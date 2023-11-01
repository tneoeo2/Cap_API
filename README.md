# Cap_API
알파벳 이미지 캡차서버  
- 알파벳 6자리로 구성된 캡차이미지를 인식하여 결과값을 반환합니다.
<br>*이미지 형식은 .png 만 지원*

### Dependency
```
python 3.8
```

### Getting Started
```
pip install -r requirements.txt 
python main.py
```

### 폴더 구조
```
│  main.py     //API 서버 실행 
│  README.md
│
├─models        //모델폴더
│  └─bw_captcha_model
│
├─utils
  │  img_module.py  //이미지 조작용 
  │  img_processing.py   //이미지 전처리
  └─ load_model.py    //모델 불러오기
```



