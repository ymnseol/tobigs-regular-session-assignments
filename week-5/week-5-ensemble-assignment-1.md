# Ensemble

## Bias & Variance
### Meaning
#### Bias(편향)
$$E[\hat{f}] - f$$
($\hat{f}$: 예측, $f$: 실제) \
실제값으로부터 예측값의 평균이 **얼마나 벗어나있는지**를 의미합니다.
#### Variance(분산)
$$E[(\hat{f} - E[\hat{f}])^2]$$
예측값들이 예측값의 평균을 기준으로 **얼마나 퍼져있는지**를 의미합니다. \
비슷한 입력에 대해 출력이 일관적일수록 (비슷한 입력에 대해 출력이 덜 퍼져있으니까) variance가 작습니다.

![bias-vs-variance](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*ObXgVrI_p2KAnflmPWUPtA.jpeg)

### Tradeoff

Bias와 variance 모두 작으면 좋겠지만, 둘은 tradeoff 관계이기 때문에 둘을 동시에 한없이 줄이기 어렵습니다.

![bias-variance-tradeoff](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*8sV6Sr9uc0Ef39YBivLzrw.jpeg)
**💡 Bias는 크면, 답을 잘 맞히지 못하는 underfitting 가능성이 높습니다. (모델이 풀고자하는 문제에 비해 너무 단순합니다.)**

**💡 Variance가 크면, 비슷한 입력값이라도 약간의 차이에 따라 예측값이 급격하게 달라지는 overfitting 가능성이 높습니다. (모델이 풀고자하는 문제에 비해 너무 복잡합니다.)**
> 비슷한 입력에 대해 출력이 일관적일수록 (비슷한 입력에 대해 출력이 덜 퍼져있으니까) variance가 작습니다.

Bias와 variance가 적절한 균형을 이룰 때의 모델을 선택하여 일반화 성능과 정확도를 적절히 챙길 수 있을 것입니다!

## Bagging(Bootstrapping Aggregating)
> **Bootstrapping**
>
> 복원추출을 사용하는 테스트, 평가지표 등을 통틀어 bootstrapping이라고 합니다.

### Method
Bagging은

1. 학습 데이터의 개수가 고정되어 있을 때(추가로 데이터를 수집한다거나 하는 상황을 배제)
2. 학습 데이터를 bootstrapping을 통해 **여러 표본데이터셋**으로 구성해서
3. 각 데이터셋으로 학습된 **여러 개의 모델**을 만들고
4. 그렇게 나온 여러 출력을 수합(평균내기, 추가 classifier 쓰기 등)하여 최종 출력을 결정하는

방법입니다.

### Objectives
Bagging의 목적은 데이터의 다양성을 확보하여 일반화 성능을 높이는 것입니다.

**💡 데이터의 다양성을 확보하여 일반화 성능을 높이기 위한 bagging은 variance를 낮추는 방법이라고 할 수 있습니다!**


## Boosting
### Method
Boosting은

0. 약한 모델을 sequential하게 연결해,
1. 이전의 모델이 약한 데이터에 대해 가중치를 부여하여
2. 해당 부분을 보강하는 방향으로 학습하는

방법입니다.

### Objectives
Boosting의 목적은 오차를 줄이는 것입니다.

**💡 모델이 잘 맞히지 못하는 부분을 더 잘 맞히도록 하기 위한 boosting은 bias를 낮추는 방법이라고 할 수 있습니다!**

## Bagging vs. Boosting

![bagging-vs-boosting](https://images.datacamp.com/image/upload/f_auto,q_auto:best/v1542651255/image_2_pu8tu6.png)
$$