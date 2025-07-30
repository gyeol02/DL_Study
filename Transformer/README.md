# Transformer ([논문 링크](https://arxiv.org/pdf/1706.03762))

## 1. 개요

Transformer는 2017년 Google Brain 팀이 발표한 자연어 처리 모델로, 기존 RNN 계열 seq2seq 구조를 완전히 대체할 수 있는 전 어텐션 기반 구조를 제안했습니다.

RNN, LSTM 없이도 번역, 요약 등 다양한 작업에서 높은 성능을 달성하며 병렬화와 장기 의존성 학습 문제를 동시에 해결했습니다.
- 핵심:
    - Self-Attention
    - Multi-Head Attention
    - Positional Encoding
    - Residual Connection + LayerNorm

---

## 2. 핵심 구조

- Encoder-Decoder 형식


---

## 3. Encoder 구조



---

## 4. Decoder 구조


---

## 5. 디렉토리 구조
```
UNet/
├── models/
│   └── Transformer.py             # UNet 모델 구현
│   └── Encoder_Decoder.py
│   └── Attension.py
