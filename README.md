# Domain-Discriminator-BERT
✏ Domain Discriminator 구현 repo

```bash
├── Chunker.py
├── layers_.py
├── modeling.py
├── mdelings2.py
├── layers_.py
├── utils.py
├── mrc_utils.py
├── optimization.py
├── optimization_.py
├── myelynn_main.py
├── integrated_main.py
├── get_dataset.py
└── vocab_bigbird.txt
``` 

## 실험내용
Pretrained KoBigBird Model을 KorQuAD 2.0, Korwiki, office, spec, law 네 가지 도메인으로 Fine-Tuning함
이후 각 도메인으로 교차평가 진행 및 Unknown Domain인 Hub 행정 dev dataset으로 평가함

## 실험 모델 구조
![discriminator](https://user-images.githubusercontent.com/64192139/212037171-e5b06a63-3f28-4192-a458-24fb77b5e249.png)

## 데이터셋
|Dataset|trained Data|Test Data|
|----|-----|------|
|KorQuAD 2.0|12655|1618|
|KorWIKI|10000|11214|
|Office|6826|1397|
|Spec|2014|300|
|Law|1546|300|
|Hub 행정|-|6979|

## 실험환경
batch_size = 4, epoch = 데이터별 2 ~ 11 epoch, learning_rate = 3e-5

## 실험결과
No Discriminator - Base
|Dataset|F1|EM|
|----|----|----|
|KorQuAD 2.0|85.17|79.23|
|KorWIKI|93.54|88.96|
|Office|82.07|72.29|
|Spec|76.52|63.33|
|Law|74.15|38.33|
|Hub 행정|89.17|83.69|

Discriminator
|Dataset|F1|EM|
|----|----|----|
|KorQuAD 2.0|86.37|80.65|
|KorWIKI|93.66|89.11|
|Office|81.66|71.79|
|Spec|86.10|78.33|
|Law|74.12|38.66|
|Hub 행정 (*Unknown Data_Not Trained*)</span>|88.95|82.96| 
