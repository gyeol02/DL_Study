# MobileNet (V1)([논문 링크](https://arxiv.org/pdf/1704.04861))

## 1. 개요

MobileNet은 모바일 및 임베디드 장치처럼 **연산 리소스가 제한된 환경**에서도 효율적으로 동작할 수 있도록 설계된 **경량 CNN 모델**입니다.  

표준 convolution을 **Depthwise Separable Convolution**으로 대체함으로써, **연산량**과 **모델 파라미터 수**를 크게 줄이는 것을 목표로 하고 있습니다.

---

## 2. 핵심 구조

- **Depthwise Separable Convolution**  
  → Standard convolution을 **Depthwise**와 **Pointwise**로 분리하여 연산량 절감  
  → 이 구조는 ResNet 등에서 사용된 1×1 convolution(`Pointwise`)과 비슷하나, 더 **적은 계산량**을 유도

- **1x1 Conv로 채널 수 줄이기**  
  → 계산 효율을 높이기 위해, feature map의 채널 수를 줄이고 다시 늘리는 구조 사용

- **모듈화된 구조 반복**  
  → 동일한 Block (`Depthwise + Pointwise`)을 여러 번 반복하여 성능 확보

---

## 3. 모델 구조

MobileNet V1 구조는 다음과 같습니다:
- Input (224 X 224)
- Conv2D (3 X 3, Stride=2)
- [Depthwise + Pointwise] x N
- GlobalAvgPool, FC Layer, Sodftmax

---

## 4. 디렉토리 구조
```
MobileNet/
├── config/
│   └── config.json             # 실험 설정값을 json으로 정의
├── models/                     
│   └── mobilenet.py            # Mobilenet 구현
├── scripts/
│   └── train_val_test.py       # Trainer & Validator & Testor 정의
├── dataset/
│   ├── data/                   # 다운로드 데이터가 저장되는 디렉토리 (CIFAR10, ...)
│   └── loader.py               # dataset_loader 함수 정의 (train/val/test 로더 생성)
├── main.py                     # main 