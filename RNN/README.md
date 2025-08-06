# RNN ([논문 링크](https://arxiv.org/abs/1808.03314))

## 1. 개요

기존의 **Multi Layer Perceptron** 은 입력과 출력이 고정된 길이를 가지며 시간적 순서나 문맥 정보를 고려하지 못하는 한계가 있었습니다.  
하지만 자연어, 음성, 시계열 데이터는 **순차적 특성**을 가지며 현재 시점의 의미가 과거 시점의 정보와 강하게 연결되어 있습니다.

이러한 연속적인 데이터의 의존성을 모델링하기 위해 **Recurrent Neural Network** 이 제안되었습니다.
RNN은 은닉 상태(hidden state)를 이용해 **이전 시점의 정보를 현재 시점 계산에 반영**함으로써 순차 데이터의 문맥(Context)을 유지할 수 있습니다.

---

## 2. RNN의 특징

- **순차 데이터 처리**: 입력 데이터를 시간 순서대로 처리하며, 이전 단계의 출력이 다음 단계 입력에 영향
- **은닉 상태**: 각 시점의 상태를 벡터로 저장하여 과거 정보 유지
- **파라미터 공유**: 모든 시점에서 동일한 가중치를 사용하므로 모델 크기 효율적
- **단점**:
  - 긴 시퀀스에서 **Vanishing/Exploding Gradient** 문제 발생
  - 병렬화가 어려워 학습 속도가 느림
  - 장기 의존성(Long-Term Dependency) 학습 한계 → LSTM, GRU 등장 배경

---

## 4. 사용법 및 설정

- config/config.json 파일에서 파러미터 조절하여 다양한 버전을 학습 가능
- 학습, 모델 정의, 데이터 로딩 모듈화
- 데이터셋: IMDB

---

## 5. 디렉토리 구조
```
RNN/
├── config/
│   └── config.json         # 모델 및 학습 설정
├── dataset/
│   └── loader.py           # 데이터셋 로드
├── models/
│   └── rnn.py              # RNN 모델 구현
├── scripts/
│   └── train_val_test.py   # Trainer & Validator & Testor 정의
├── utils/
│   └── preprocess.py       # text 전처리
├── main.py                 # main
```
