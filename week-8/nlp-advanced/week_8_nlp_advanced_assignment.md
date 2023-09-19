# Improving Language Understanding by Generative Pre-Training (GPT-1)

### Before reading

**논문 제목을 보고, 해당 모델이 어떤 방법론을 바탕으로 할지 가설을 세워봅시다.**

- NLU task를 ***generative*** pre-training을 통해 해결하는 새로운 방법론을 제시하지 않을까?
- Generative Pre-Training: Language Understanding을 Generative Pre-Training으로 개선한다… Generative model은 데이터로부터 확률분포를 학습하고, 확률분포를 따르는 sample을 ***생성***하는 것으로 알고 있다.
    
    <aside>
    💡 **Generative model**
    
    - Generation: 입력 데이터의 확률분포 $p(x)$를 학습하고, 그러한 확률분포를 따르는 sample 생성
    - Density estimation: 판별하고자 하는 데이터를 입력했을 때, 확률값이 높게 나오면 맞는 데이터
    - Explicit model: $P(x)$확률 분포가 값이 항상 0보다 크고 모든 가능한 값들로 적분했을 때 1이 된다면 이를 proper distribution이라고 부르는데, 이를 만들어내는 모델을 Explicit model이라고 한다.
    </aside>
    
- 왜 Generative Pre-training을 할 생각을 했을까?
    
    → Unlabeled data는 많지만, 특정 task를 위한 labeled data는 부족해서, discriminatively trained model이 잘 작동하기 어려운 상황이다.
    
    → **Unlabeled data를 통해 generative pre-training을 수행해서 데이터에 대한 확률분포를 충분히 학습하고, 각 task에 대해 discriminative fine-tuning을 하자!**
    
    > **Abstract**
    > 
    > 
    > *Although large unlabeled text corpora are abundant, labeled data for learning these specific tasks is scarce, making it challenging for discriminatively trained models to perform adequately. We demonstrate that large gains on these tasks can be realized by generative pre-training of a language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task.*
    > 
- GPT는 Transformer의 decoder를 바탕으로 한다고 하는데, Transformer의 decoder는 encoding된 입력 문장의 정보와 이전에 생성한 단어의 정보를 활용해 다음 단어를 생성한다.
- 이 논문이 Language Understanding에서 Generative pre-training을 처음으로 사용한 건가?
    
    → 이전에도 여러 시도가 있었나본데(이건 generative pre-training을 사용한 시도들을 말하는 건지 아닌지 찾아보아야 함), 여기서는 이전 시도와 다르게 fine-tuning 과정에서 task-aware input을 사용해서 모델 구조 변형을 최소화하면서 효과적으로 transfer learning을 하고자 한다.
    
    > **Abstract**
    > 
    > 
    > *In contrast to previous approaches, we make use of task-aware input transformations during fine-tuning to achieve effective transfer while requiring minimal changes to the model architecture.*
    > 

**모델의 메인 figure를 보고 감을 잡아봅시다.**

<img width="1337" alt="gpt_figure" src="https://github.com/ymnseol/tobigs-regular-session-assignments/assets/105059564/9089b3fd-6fc8-4d5f-bc22-547a50e26d56">


- Abstract에서 각 task를 위해 모델을 사용한다고 해도 기존 모델의 구조를 최소한으로만 변형할 수 있도록 했다고 했는데, 그 말대로 task에 맞게 input transformation은 다르게 하되 Transformer 구조는 그대로 사용한다.
- 풍부한 unlabeled data를 이용해 generative pre-training을 해서 강력한 모델을 만들고, 이후 다양한 language understaning task에서 이 모델을 fine-tuning해 사용할 수 있도록 한다!

**해당 모델을 구현한 코드가 있는지 체크해봅시다.**

https://github.com/openai/finetune-transformer-lm

**이 논문에서 풀고자 하는 문제가 무엇인가요? Task를 정의해봅시다.**

- Labeled data가 적은 language understanding task에도 모델이 잘 작동하게 하고자 한다.
