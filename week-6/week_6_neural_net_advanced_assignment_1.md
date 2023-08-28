# Neural Network

## Learning Rate Scheduler

### Cosine Annealing
$$
\eta_t = \eta_{min} + \frac{1}{2} (\eta_{max} - \eta_{min}) \bigg( 1 + \cos\bigg({\frac{T_{cur}}{T_{max}}\pi}\bigg) \bigg)
$$

* $\eta_{t}$:  step t에서의 learning rate
* $\eta_{max}$: learning rate 최댓값 (hyperparameter)
  * 학습 시 최대 learning rate는 $\eta_{max}$입니다.
  * (Warmup 등을 하지 않으면) initial learning rate와 같습니다.
* $\eta_{min}$: learning rate 최솟값 (hyperparameter)
  * 학습시 learning rate는 $\eta_{min}$ 아래로 감소하지 않습니다.
* $T_{max}$: 감소 / 증가 주기 (반주기)

![cosing-annealing](https://gaussian37.github.io/assets/img/dl/pytorch/lr_scheduler/7.png)

### Cosine Annealing with Warm Restarts

$$
\eta_t = \eta_{min} + \frac{1}{2} (\eta_{max} - \eta_{min}) \bigg( 1 + \cos\bigg({\frac{T_{cur}}{T_{i}}\pi}\bigg) \bigg)
$$

* $\eta_{t}$:  step t에서의 learning rate
* $\eta_{max}$: learning rate 최댓값 (hyperparameter)
  * 학습 시 최대 learning rate는 $\eta_{max}$입니다.
  * (시작 시 warmup 등을 하지 않으면) initial learning rate와 같습니다.
* $\eta_{min}$: learning rate 최솟값 (hyperparameter)
  * 학습시 learning rate는 $\eta_{min}$ 아래로 감소하지 않습니다.
* $T_{i}$: 주기

Learning rate가 감소하다가 현재 epoch $T_{cur}$와 $T_i$가 같아지는 경우, learning rate는 $\eta_{min}$이 됩니다. 이후 $T_{cur}$를 0으로, learning rate $\eta_{t}$를 $\eta_{max}$로 초기화합니다.

초기에 $\eta_{min}$에서 시작하여 learning rate를 증가시킨 뒤 (warmup) 주기를 따르도록 할 수도 있고, 주기를 학습이 진행될수록 길게 설정할 수도 있습니다.

## Optimizer

> *🥲 과제를 잘못 읽고 optimizer에 대해 설명을 작성했습니다...*

### Momentum
> SGD(Stochastic Gradient Descent)에 **관성 개념 추가해 step 방향 조정**하기

$$
v_{t+1} = \mu \cdot v_{t} + g_{t+1}
$$
$$
p_{t+1} = p_t - \eta \cdot v_{t+1}
$$

* $v$: velocity
* $\mu$: momentum
  * 이전의 이동을 얼마나 반영할지를 나타냅니다.
  * 0인 경우 SGD와 동일합니다.
* $g$: gradient
* $p$: parameters
* $\eta$: learning rate

Gradient를 업데이트할 때, 전의 gradient의 방향을 어느 정도(momentum $\beta$의 크기만큼) 유지하도록 합니다. 이전 gradient를 **관성**과 같이 적용시키는 방법입니다.

**💡 Momentum은 지난 mini-batch와는 다른 이번 mini-batch에 대한 학습에서도, 이전 mini-batch의 정보 역시 활용하도록 합니다.**

👍 급격한 방향의 변화를 $\mu \cdot v_{t}$가 완화해주기 때문에, SGD에 비해 수렴 속도가 빠를 수 있습니다.

👎 Velocity가 gradient보다 큰 경우, 관성에 의해 최적의 지점을 지나쳐버리는 현상이 발생할 수 있습니다.

### Adagrad(Adaptive Gradient)
> SGD에서 **loss function의 변화에 따라 학습률을 결정**하여 반영하기

$$
G_{t+1} = G_{t} + (g_{t+1})^2
$$
$$
p_{t+1} = p_{t} - \frac{\eta}{\sqrt{G_{t+1}} + \epsilon} g_{t+1}
$$

* $p$: parameters
* $\eta$: initial learning rate
* $g$: gradient
* $G$: $g^2$의 합
* $\epsilon$: 분모가 0이 되지 않게 하기 위한 값

Parameter가 얼마나 변했는지를 따져, 많이 변화된 parameter의 경우 더 적게, 적게 변화된 parameter의 경우 더 많이 변화시키도록 learning rate를 변화시킵니다.

👍 Learning rate가 적합하지 않아 발생하는 문제(너무 크면 발산하는 등)를 learning rate를 조정하여 해결할 수 있습니다.

👎 Gradient의 제곱의 합 $G$는 학습을 할수록 커져서, **learning rate이 0에 수렴해 학습이 멈추는 현상**이 발생할 수 있습니다.

### RMSProp(Root Mean Square Propagation)
> Adagrad에서 지수가중이동평균을 이용해, 최신의 기울기를 더 크게 반영하도록 합니다.

$$
G_{t+1} = \alpha \cdot G_{t} - (1 - \alpha) (g_{t+1})^2
$$
$$
p_{t+1} = p_{t} - \frac{\eta}{\sqrt{G_{t+1}} + \epsilon} g_{t+1}
$$

* $G$: gradient의 지수가중이동평균
* $p$: parameters
* $\eta$: initial learning rate
* $g$: gradient
* $\epsilon$: 분모가 0이 되지 않게 하기 위한 값

$G$를 계산할 때, 최근의 gradient를 더 크게 반영하도록 합니다.

👍 Adagrad에서 발생했던 학습이 멈추는 현상을 완화할 수 있습니다.

### Adam(Adaptive Moment Estimation)
> Momentum + RMSProp

$$
m_{t+1} = \beta_{1} \cdot m_{t} + (1 - \beta_{1}) g_{t}
$$
$$
v_{t+1} = \beta_{2} \cdot v_{t} + (1 - \beta_2) (g_{t+1})^2
$$
$$
\hat{m}_{t+1} = \frac{m_{t+1}}{1 - (\beta_{1})^{t+1}} \\
$$
$$
\hat{v}_{t+1} = \frac{v_{t+1}}{1 - (\beta_{2})^{t+1}}
$$
$$
p_{t+1} = p_{t} - \eta \frac{\hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1}} + \epsilon}
$$

* $m$: first moment
* $v$: second moment
* $\hat{m}$: bias-corrected first moment
  * Momentum의 velocity 아이디어에 지수가중이동평균 개념을 적용합니다.
* $\hat{v}$: bias-corrected second moment
  * RMSProp의 gradient의 제곱의 지수가중이동평균 $G$ 아이디어를 적용합니다.
* $\beta_{1}$: first moment의 decay rate
* $\beta_{2}$: second moment의 decay rate
* $\eta$: learning rate
* $\epsilon$: 분모가 0이 되지 않게 하기 위한 값
* $p$: parameters

학습을 시작할 때,

* $m_0 = 0$
* $v_0 = 0$
* $\beta_{1} = 0.9$
* $\beta_{2} = 0.999$

로 초기화되어있습니다.

그러면,

$$
m_{1} \\
= \beta_{1} \cdot m_{0} + (1 - \beta_{1}) g_{1} \\
= 0 + 0.1 \cdot g_{1}
$$
$$
v_{1} \\
= \beta_{2} \cdot v_{0} + (1 - \beta_2) (g_{1})^2 \\
= 0 + 0.001 \cdot (g_{1})^{2}
$$
로, moment가 0에 너무 가깝도록 치우쳐진채로 시작하게 됩니다.

$\beta_{1}$, $\beta_{2}$는 0 이상 1 미만인 실수값으로, bias-corrected first moment $\hat{m}$과 bias-corrected second moment $\hat{v}$의 분모는 값이 점점 커져 1에 가까워집니다. 학습 초반에 bias correction을 어느 정도 수행하고 나면, bias correction은 효과가 미미해지게 됩니다. (Bias correction을 적용하는 것과 적용하지 않는 것이 비슷해집니다.)

**💡 Adam은 Momentum의 velocity와 RMSProp의 지수가중이동평균, bias correction을 통해 처음부터 적절한 값으로 학습을 잘 시작하도록 하고, step 방향 및 사이즈 역시 안정적으로 학습에 반영할 수 있도록 합니다.**

## Generalization Performance

### Ensemble

> **Bagging(Bootstrapping Aggregating)**
> > **Bootstrapping**
> >
> > 복원추출을 사용하는 테스트, 평가지표 등을 통틀어 bootstrapping이라고 합니다.
>
> **Method**
> 
> Bagging은
> 1. 학습 데이터의 개수가 고정되어 있을 때(추가로 데이터를 수집한다거나 하는 상황을 배제)
> 2. 학습 데이터를 bootstrapping을 통해 **여러 표본데이터셋**으로 구성해서
> 3. 각 데이터셋으로 학습된 **여러 개의 모델**을 만들고
> 4. 그렇게 나온 여러 출력을 수합(평균내기, 추가 classifier 쓰기 등)하여 최종 출력을 결정하는
> 
> 방법입니다.
> 
> **Objectives**
> 
> Bagging의 목적은 데이터의 다양성을 확보하여 일반화 성능을 높이는 것입니다.
> 
> **💡 데이터의 다양성을 확보하여 일반화 성능을 높이기 위한 bagging은 variance를 낮추는 방법이라고 할 수 있습니다!**

### Early Stopping

Validation loss 또는 다른 metric을 참고하여, 초기에 설정한 학습 epoch / steps 보다 더 빨리, 모델이 overfitting되기 전 학습을 멈출 수 있습니다.

```Python
...
if val_loss >= best_loss:
    current_patience += 1
    if current_patience >= patience:
        break
else:
    best_loss = val_loss
    current_patience = 0
...
```

### Label Smoothing

특히 분류 문제에서는, 모델이 ground truth로 분류할 확률을 1에 가깝게 도출하도록 학습합니다. 이러한 경우, 모델이 지나친 확신을 가지게 될 수 있습니다.(학습 데이터의 분포에만 과도하게 일치하는 분포를 학습할 수 있습니다.)

학습 데이터의 정답 label이 아닌 다른 label에도 일정한 확률을 부여한 분포를 학습시키도록 할 수 있습니다.

## References

[Pytorch Learning Rate Scheduler (러닝 레이트 스케쥴러) 정리](https://gaussian37.github.io/dl-pytorch-lr_scheduler/#cosineannealinglr-1)

[Why is it important to include a bias correction term for the Adam optimizer for Deep Learning?](https://stats.stackexchange.com/questions/232741/why-is-it-important-to-include-a-bias-correction-term-for-the-adam-optimizer-for)

[딥러닝 용어정리, RMSProp, Adam 설명](https://light-tree.tistory.com/141)

[Optimization(최적화 알고리즘) : Mini-batch/Momentum/RMSprop/Adam](https://junstar92.tistory.com/81)

[ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION](https://arxiv.org/pdf/1412.6980v9.pdf)

[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf)
