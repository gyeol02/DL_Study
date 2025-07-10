# VGGNet ([논문 링크](https://arxiv.org/pdf/1409.1556))

## 1. 개요

VGGNet은 딥러닝에서 네트워크의 깊이(depth)가 이미지 인식 정확도에 어떤 영향을 주는지를 체계적으로 분석한 연구입니다. 당시 ConvNet 구조가 일반화되던 상황에서, VGG는 구조를 단순하게 유지한 채 3×3의 작은 필터를 반복적으로 쌓는 방식으로 깊이를 최대 19층까지 늘렸습니다.

이러한 접근은 복잡한 연산 없이도 모델의 표현력을 극대화할 수 있다는 점을 증명하였고, ImageNet 2014 대회에서 분류와 위치 추적 부문에서 각각 2위와 1위를 차지하며 그 성능을 입증했습니다. 또한, VGGNet은 다른 데이터셋에서도 단순한 분류기(classifier)만 붙여도 우수한 성능을 보였습니다.

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

- 3x3 Conv → BatchNorm → ReLU 로 구성
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
