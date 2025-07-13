# DenseNet ([논문 링크](https://arxiv.org/pdf/1608.06993))

## 1. 개요

기존의 딥러닝 CNN 아키텍쳐들이 **네트워크의 depth를 증가**시켜 성능을 높이려는 시도를 지속했지만,
이는 **Gradient Vanishing, Feature Reuse 부족, 파러미터 수 과다 등의 한계**를 가졌습니다.

이러한 문제를 완화하기 위해 **ResNet**의 **Skip Connection**을 통해 **identity** 정보를 더하여 학습의 안정성과 정보 보존에 도움을 주었습니다.
하지만 이 또한 바로 이전 레이어의 출력만 전달하여 **Feature Reuse에 제한적**이라는 단점이 있습니다.

그래서 만들어진 **DenseNet**은 이전 레이어의 출력만을 연결하는 ResNet과 다르게 **이전의 모든 레이어 출력과 연결**되어 입력으로 사용하는 구조입니다.
각 레이어는 다음과 같이 정의 됩니다.

```
x_l = H_l([x_0, x_1, …, x_{l-1}])
```

---

## 2. Dense_Layer & Dense_Block 구조

### Dense_Layer

- 구조: **BN → ReLU → 1×1 Conv → BN → ReLU → 3×3 Conv**
- 역할: 새로운 Feature Map 'k'개(Growth Rate)를 생성하고 입력과 'concat'하여 전달

### Dense_Block

- 구조: 여러 개의 Dense_Layer로 구성
- 역할: 채널 수 누적 -> out_ch = in_ch + k × num_layers

---

## 3. Transition_Layer

- 구조: **BN → ReLU → 1×1 Conv → 2×2 AvgPool**
- 역할: Feature Map 크기 및 채널 수 감소 (Compression)

---

## 4. 사용법 및 설정

- config/config.json 파일에서 모델명을 "densenet121", "densenet169", "densenet201", "densenet264"로 바꾸어 다양한 버전을 학습 가능
- 학습, 모델 정의, 데이터 로딩 모듈화

---

## 5. 디렉토리 구조
```
DenseNet/
├── config/
│   └── config.json         # 모델 및 학습 설정
├── dataset/
│   └── loader.py           # 데이터셋 로딩 및 전처리
├── models/
│   └── densenet.py         # DenseNet 모델 구현
├── scripts/
│   └── main.py             # 학습 및 테스트 실행
```
