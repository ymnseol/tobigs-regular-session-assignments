# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)

## Before reading

**논문 제목을 보고, 해당 모델이 어떤 방법론을 바탕으로 할지 가설을 세워봅시다.**

- NLP에서 처음으로 제시했던 Transformer가 문장을 이루는 token들의 attention 연산을 수행했던 것에서 착안해, 이미지를 어떠한 구성 단위로 나누어 마치 NLP의 token처럼 attention 연산을 수행할 것

- 그렇다면 CNN과의 차이는 무엇일까? CNN도 이미지를 filter 크기만큼 조각조각으로 보는 것이 아닌가?
    - Inductive bias: CNN > Transformer

        > **Inductive bias**
        > 
        > 학습할 때 보지 못했던 데이터에 대해서도 정확한 예측을 하기 위해 사용하는 추가적인 가정

    - 이미지의 경우 서로 가까이 있는 pixel들끼리 관계가 있다는 것을 알고 있기에, 이러한 부분을 가정하는 CNN(kernel로 가까이 있는 픽셀들끼리 연산)은 inductive bias가 상대적으로 높다고 할 수 있다.
    - Transformer의 경우, 모든 head가 일관적으로 같은 부분(e.g. 주변 pixel)을 강조하여 보지는 않는다. 어떤 head는 뒤의 layer에서도 모든 patch를 골고루 볼 수도 있고, 어떤 head는 주변 patch를 보도록 한다. 우리는 multi-head attention으로 다양한 관점에서 데이터를 보도록 하고, 그 관점들은 우리가 사전에 정의하지 않는다!

**모델의 메인 figure를 보고 감을 잡아봅시다.**

<img width="971" alt="vit_figure" src="https://github.com/ymnseol/tobigs-regular-session-assignments/assets/105059564/0ef423cb-ac9e-497b-b638-e51c86220abf">

- 입력
  - 문장을 token 단위로 나누는 것과 마찬가지로, 이미지를 patch 단위로 나누어 모델에 입력으로 줌
  - 문장에서 [CLS] token을 제일 앞에 추가해줘서 문장 정보를 추출하고자 할 때 사용했던 것과 마찬가지로, 이미지 역시 [CLS]를 추가해 줌

- Transformer Encoder
  - 정말로 original Transformer의 encoder block과 세부 layer 순서만 조금 다르고 거의 유사한 encoder block으로 이루어져 있음

**해당 모델을 구현한 코드가 있는지 체크해봅시다.**

https://github.com/google-research/vision_transformer
