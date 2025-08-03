# Transformer ([논문 링크](https://arxiv.org/pdf/1706.03762))

## 1. 개요

Transformer는 2017년 Vaswani et al.이 발표한 "Attention is All You Need" 논문에서 제안된 모델로, 기존의 순환 신경망(RNN) 없이 순수 Attention 메커니즘만으로 문장 간 의미를 학습합니다.

이 구조는 특히 병렬화가 가능하고 장기 의존성 문제를 극복할 수 있어, 자연어 처리뿐만 아니라 이미지, 음성 등 다양한 도메인으로 확장되었습니다.

> **핵심 개념**  
> - **Self-Attention**: 단일 문장 내 단어 간 관계 파악  
> - **Multi-Head Attention**: 서로 다른 시점에서의 관계를 병렬적으로 학습  
> - **Positional Encoding**: 순서를 알 수 없는 구조의 한계를 극복하기 위한 위치 정보 인코딩  
> - **Residual Connection + Layer Normalization**: 학습 안정성과 성능 향상을 위한 구조적 요소

---

## 2. 핵심 구조

Transformer는 **Encoder-Decoder** 구조로 구성됩니다.

- **Encoder**: 입력 문장의 의미를 고차원 벡터로 추상화 (총 N개 블록 반복)
- **Decoder**: 추상화된 정보를 기반으로 출력 문장 생성 (총 N개 블록 반복)

각 블록은 다음으로 구성됩니다:

| 블록 구성       | 세부 구조 |
|----------------|-----------|
| Encoder Layer  | Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm |
| Decoder Layer  | Masked Multi-Head Attention → Add & Norm → Encoder-Decoder Attention → Add & Norm → Feed Forward → Add & Norm |

---

## 3. Encoder 구조

- 입력 임베딩 + 위치 인코딩
- 여러 개의 **Encoder Layer**로 구성 (기본 6개)
- 각 Layer는 다음 순서로 구성됨:
    1. **Multi-Head Self-Attention**
    2. **Residual Connection + LayerNorm**
    3. **Feed Forward Network (FFN)**
    4. **Residual Connection + LayerNorm**

> Self-Attention은 입력 내 각 단어가 다른 모든 단어와의 관계를 고려하여 자신을 표현

---

## 4. Decoder 구조

- 출력 임베딩 + 위치 인코딩
- 여러 개의 **Decoder Layer**로 구성 (기본 6개)
- 각 Layer는 다음 순서로 구성됨:
    1. **Masked Multi-Head Self-Attention**
    2. **Residual Connection + LayerNorm**
    3. **Encoder-Decoder Attention** (Encoder 출력을 Query로 사용)
    4. **Residual Connection + LayerNorm**
    5. **Feed Forward Network**
    6. **Residual Connection + LayerNorm**

> Masked Attention은 미래 단어를 보는 치팅을 방지하여, autoregressive decoding을 가능하게 함

---

## 5. 디렉토리 구조
```
Transformer/
├── config/
│   └── config.json             # 실험 설정값을 json으로 정의
├── models/                     
│   └── Transformer.py          # Transformer Class 구현
│   └── Encoder_Decoder.py      # Encoder, Decoder 구현
│   └── Attension.py            # Scaled Dot-Product, Multi-Head Attention 등 Attention 모듈
├── utils/
│   └── mask.py                 # padding mask, subsequent mask, combined mask 함수 정의
├── scripts/
│   └── train_val_test.py       # Trainer & Validator & Testor 정의
├── dataset/
│   ├── data/                   # HuggingFace 다운로드 데이터가 저장되는 디렉토리
│   └── loader.py               # dataset_loader 함수 정의 (train/val/test 로더 생성)
├── utils/
│   └── mask.py
│   └── scheduler.py            # Learning rate scheduler 
├── main.py                     # main 
├── test.py                     # test 
```