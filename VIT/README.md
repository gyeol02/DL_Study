# ViT ([논문 링크](https://arxiv.org/pdf/2010.11929))

## 1. 개요

기존의 컴퓨터 비전 모델은 주로 CNN(Convolutional Neural Network)을 사용해 왔습니다.  
하지만 NLP 분야에서 Transformer가 강력한 성능을 보이자, 이미지 처리에도 Transformer를 적용하려는 시도를 했고. 2020년 Google Research의 "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" 논문에서 제안된 **Vision Transformer**는 이미지를 **패치 단위 토큰**으로 변환해 Transformer Encoder에 입력함으로써, **전역적 문맥 정보**를 더 효율적으로 학습할 수 있음을 보여주었습니다. 

주요 장점은 다음과 같습니다:
- Self-Attention 기반의 관계 학습 -> CNN의 Receptive field 한계 극복
- 대규모 데이터셋에서 뛰어난 성능
- 단순한 구조로 NLP와 Vision에서 아케틱처 공유 가능

---

## 2. 핵심 구조

핵심 구조는 총 3구간으로 구성되며 이는 다음과 같습니다:

1. **Patch Embedding**  
   - 이미지를 작은 패치(예: 16×16)로 나눈 뒤, 각각을 Flatten → Linear Projection으로 고정된 차원 벡터로 변환
   - Transformer 입력과 동일한 형태(시퀀스)로 맞추기 위해 CLS 토큰과 위치 임베딩(Position Embedding)을 추가

2. **Transformer Encoder**  
   - NLP의 Transformer Encoder와 동일한 구조.
   - 여러 개의 **Multi-Head Self-Attention** 블록과 **Feed Forward Network(MLP)**로 구성
   - 각 블록은 **LayerNorm + Residual Connection**을 포함하여 안정적 학습을 보장

3. **Classification Head**  
   - 첫 번째 토큰(CLS 토큰)의 최종 출력만 사용하여 클래스 예측

---

## 3. 구성 요소

- **CLS Token**: 문장의 대표 의미를 담는 BERT의 [CLS] 토큰과 동일한 개념. 
  ViT에서는 이미지 전체의 정보를 모으는 대표 토큰 역할
  
- **Position Embedding**: 순서 정보가 없는 Self-Attention에 위치 정보를 제공
  Patch Embedding에 더해져 각 패치가 원래 이미지의 어느 위치였는지 알려줌

- **Multi-Head Self-Attention**  
  - 입력을 Query(Q), Key(K), Value(V)로 변환 후, QK^T로 유사도를 구해 Value를 가중합
  - 여러 Head를 병렬로 사용하여 서로 다른 서브공간에서의 관계를 동시에 학습

- **Feed Forward Network**  
  - 비선형 변환(GELU 등)으로 특성 변환
  - 각 토큰 벡터별로 독립적으로 적용

- **LayerNorm + Residual Connection**  
  - 학습 안정화 및 그라디언트 흐름 보존

---

## (추가) Weight Intialization

ViT는 Transformer 계열과 동일하게 **안정적 학습**을 위해 초기화가 매우 중요하기 때문에 본 구현에서는 **BERT 관례적 초기화**를 따랐습니다

**초기화 방식**
- **Truncated Normal**: 모든 Linear/Conv2d 가중치, CLS 토큰, Position Embedding을 평균 0, 표준편차 0.02의 절단 정규분포에서 샘플링
  - 꼬리 값을 잘라내어 과도하게 큰 초기값을 방지
  - Self-Attention의 Q/K 분산이 커져 Softmax가 편향되는 문제를 예방
- **Bias 초기값**: 0으로 설정
- **LayerNorm**: weight=1, bias=0 → 초기엔 항등변환처럼 동작하여 학습 안정성 확보

**이 초기화가 중요한 이유**
- Transformer는 깊고 잔차 연결이 많은 구조 → 작은 초기값이 안정성에 유리
- ViT에서도 동일한 이유로 BERT와 동일한 초기화가 사실상의 표준
- 큰 표준편차를 쓰면 초기 Attention 분포가 지나치게 편향되어 학습이 불안정해질 가능성


---

## 4. 디렉토리 구조
```
VIT/
├── config/
│   └── config.json             # 실험 설정값을 json으로 정의
├── models/                     
│   └── vit.py                  # VIT Class 구현
│   └── Attension.py            # Scaled Dot-Product, Multi-Head Attention 등 Attention 모듈
├── scripts/
│   └── train_val_test.py       # Trainer & Validator & Testor 정의
├── dataset/
│   └── loader.py               # dataset_loader 함수 정의 (train/val/test 로더 생성)
├── utils/
│   └── scheduler.py            # Learning rate scheduler 
├── main.py                     # main 
```