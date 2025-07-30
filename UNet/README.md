# UNet ([논문 링크](https://arxiv.org/pdf/1505.04597))

## 1. 개요

기존의 패치 단위의 CNN은 이미지 Classification에는 효과적이었지만, **픽셀 단위로 구분해야하는 Segmentation 작업에는 한계**가 존재했습니다.

따라서, **적은 양의 데이터**로도 **정확한 위치 기반 분할**이 가능한 모델이 필요했습니다.

이러한 한계를 해결하기 위한 **UNet**은 소량의 학습 데이터로도 **정확한 Segmentation을 수행**할 수 있으며, 데이터 증강을 활용해 성능을 끌어올렸습니다.

또한, **Up Sampling과 Skip Connection을 통한 세밀한 Localization이 가능**합니다.

---

## 2. 핵심 구조

UNet은 **Encoder-Decoder 구조**로 구성되며 각 단계는 다음과 같습니다.
- **Encoder**: (Conv → ReLU) X 2 → MaxPool
- **Decoder**: Up Sample → concat → Conv X 2
- **Final Layer**: 1X1 Conv (출력 채널 수 = 클래스 수)

---

## 3. Encoder 구조

- 구조: **Conv 3X3 → ReLU → Conv 3X3 → ReLU → MaxPooling(2X2)**
- 특징: 
    - 각 블록 마다 채널 수 증가(64 → 128 → 256 → ...)
    - 점점 더 깊은 특징 추출

---

## 4. Decoder 구조

- 구조: **Up Sampling → Conv 2X2**
- 특징: 
    - 이전 Encoder와 Concat → Conv 3X3 → ReLU → Conv 3X3 → ReLU
    - 해상도 복원, 채널 수 감소
---

## 5. 사용법 및 설정

- config/config.json 파일에서 이미지 crop_size, out_channels등 파러미터 조절하여 다양한 버전을 학습 가능
- 학습, 모델 정의, 데이터 로딩 모듈화
- 데이터셋: ISIC 2016 (Task1)

---

## 6. 디렉토리 구조
```
UNet/
├── config/
│   └── config.json         # 모델 및 학습 설정
├── dataset/
│   └── loader.py           # 데이터셋 로딩 및 전처리
├── models/
│   └── unet.py             # UNet 모델 구현
├── scripts/
│   └── main.py             # 학습 및 테스트 실행
├── utils/
│   └── metrics.py          # IoU 계산