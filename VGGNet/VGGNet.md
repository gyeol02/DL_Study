# VGGNet ([논문 링크](https://arxiv.org/pdf/1409.1556))

## 1. 개요

딥러닝 모델의 깊이가 깊어질수록 무조건 성능이 향상되는 것은 아닙니다.
ResNet 논문에 따르면, 56개의 레이어를 가진 네트워크가 20개짜리보다 성능이 떨어지는 경우도 있었습니다.

이 문제를 해결하기 위해 ResNet은 Residual Learning(잔차 학습) 이라는 구조를 도입합니다.
이를 통해 매우 깊은 네트워크도 안정적으로 학습할 수 있게 됩니다.

---

## 2. 핵심 구조

- Conv 레이어는 모두 3x3 필터 사용
- stride = 1, padding = 1
- ReLU 활성화 함수 사용
- 2x2 max pooling (stride=2)으로 다운샘플링
- 깊이는 A(11-layer)부터 E(19-layer)까지 확장 가능
- FC layer는 4096-4096-1000 구조 (ImageNet 기준이고 구현한 모델에서는 CIFAR10 데이터를 사용해서 4096-4096-1000 구조)

---

## 3. LayerBlock 구조

- 3x3 Conv → BatchNorm → ReLU 로 구성됨
- ResNet과는 다르게 skip connection은 없음

---

## 4. 사용법 및 설정
- config/config.json 파일에서 모델명을 "vggnet11", "vggnet13", "vggnet16", "vggnet19"로 바꾸어 다양한 버전을 학습 가능
- n_classes 인자를 통해 데이터셋 클래스 수에 맞춰 분류기 조정 가능
- 학습, 모델 정의, 데이터 로딩 모듈화

---

## 5. 디렉토리 구조
```
VGGNet/
├── config/
│   └── config.json         # 모델 및 학습 설정
├── dataset/
│   └── loader.py           # 데이터셋 로딩 및 전처리
├── models/
│   └── resnet.py           # ResNet 모델 구현
├── scripts/
│   └── main.py             # 학습 및 테스트 실행