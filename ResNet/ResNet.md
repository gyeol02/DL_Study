# ResNet ([논문 링크](https://arxiv.org/pdf/1512.03385))

## 1. 개요

딥러닝 모델의 깊이가 깊어질수록 무조건 성능이 향상되는 것은 아닙니다.
ResNet 논문에 따르면, 56개의 레이어를 가진 네트워크가 20개짜리보다 성능이 떨어지는 경우도 있었습니다.

이 문제를 해결하기 위해 ResNet은 Residual Learning(잔차 학습) 이라는 구조를 도입합니다.
이를 통해 매우 깊은 네트워크도 안정적으로 학습할 수 있게 됩니다.

---

## 2. 잔차 학습이란?

잔차 학습은 네트워크가 직접적으로 H(x)를 학습하는 대신,
입력 x와 출력 H(x)의 차이인 F(x) = H(x) - x를 학습하는 방식입니다.

결과적으로 출력은 다음과 같이 표현됩니다:

```
H(x) = F(x) + x
```

이 구조는 shortcut connection(스킵 연결)을 통해 구현되며, 입력을 다음 블록으로 건너뛰어 더하는 방식입니다.

---

## 3. Bottleneck 구조

이 프로젝트에서는 ResNet의 Bottleneck 구조를 사용합니다.

Residual block은 다음과 같이 구성됩니다:

```
1x1 conv → 3x3 conv → 1x1 conv
```

이는 파라미터 수를 줄이면서도 성능은 유지하는 구조로, ResNet-50, 101, 152에서 사용됩니다.

---

## 4. 사용법 및 설정
- config/config.json 파일에서 모델명을 "resnet50", "resnet101", "resnet152" 등으로 바꾸어 다양한 버전을 학습시킬 수 있습니다.
- 학습, 모델 정의, 데이터 로딩이 모듈화 했습니다.

---

## 5. 디렉토리 구조
```
RESNET/
├── config/
│   └── config.json         # 모델 및 학습 설정
├── dataset/
│   └── loader.py           # 데이터셋 로딩 및 전처리
├── models/
│   └── resnet.py           # ResNet 모델 구현
├── scripts/
│   └── main.py             # 학습 및 테스트 실행
```
