# Recurrent Neural Networks (RNN, LSTM, GRU) ([논문 링크](https://arxiv.org/abs/1808.03314))

## 1. 개요

순차적 특성을 가진 자연어, 음성, 시계열 데이터는 현재 시점의 의미가 과거 시점 정보와 강하게 연결되어 있습니다.
이를 모델링하기 위해 RNN 계열 모델이 사용됩니다.

RNN, LSTM, GRU 세 가지 모델을 구현하며 공통된 구조를 기반으로 각 모델의 특징에 맞게 학습과 추론이 가능합니다.

---

## 2. 공통된 특징

- **순차 데이터 처리**: 입력 데이터를 시간 순서대로 처리하며, 이전 단계의 출력이 다음 단계 입력에 영향
- **은닉 상태**: 각 시점의 상태를 벡터로 저장하여 과거 정보 유지
- **파라미터 공유**: 모든 시점에서 동일한 가중치를 사용하므로 모델 크기 효율적
- **단점**:
  - 긴 시퀀스에서 **Vanishing/Exploding Gradient** 문제 발생
  - 병렬화가 어려워 학습 속도가 느림
  - 장기 의존성(Long-Term Dependency) 학습 한계 → LSTM, GRU 등장 배경

---

## 3. 모델별 개요 및 작동 방식

### 3.1 RNN

- **개요**: 가장 기본적인 순환 신경망으로 각 시점의 은닉 상태는 이전 시점 은닉 상태와 현재 입력의 선형 결합 후 비선형 활성화(ReLU, Tanh)를 적용

- **수식**: 
$$

$$

- **특징**: 구조 단순, 구현 용이, 하지만 장기 의존성 학습이 어려움

### 3.2 LSTM

- **개요**: 장기 의존성을 학습하기 위해 **Cell State**와 **Input, Forget, Output Gate**를 추가

- **작동 방식**:
  - **Forget Gate**: 이전 정보를 얼마나 유지할지 결정
  - **Input Gate**: 현재 입력 정보를 얼마나 반영할지 결정
  - **Output Gate**: 다음 은닉 상태로 출력할 값을 결정

- **특징**: 긴 시퀀스에서도 안정적인 학습 가능 Vanishing Gradient 문제 완화

### 3.3 GRU

- **개요**: LSTM의 단순화 버전, **Update Gate**와 **Reset Gate**만 사용

- **작동 방식**:
  - **Update Gate**: 이전 상태와 현재 입력의 혼합 정도 결정
  - **Reset Gate**: 이전 은닉 상태를 얼마나 반영할지 결정

- **특징**: 특징: LSTM보다 파라미터 적음 학습 속도 빠름, 성능은 LSTM과 유사

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
│   └── lstm.py             # LSTM 모델 구현
│   └── gru.py              # GRU 모델 구현
├── scripts/
│   └── train_val_test.py   # Trainer & Validator & Testor 정의
├── utils/
│   └── preprocess.py       # text 전처리
├── main.py                 # main
```
