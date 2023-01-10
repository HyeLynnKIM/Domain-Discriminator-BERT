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
PRetrained KoBigBird Model을 Fine-Tuning하여  


|Dataset|trained Data|Test Data|
|----|-----|------|
|KorQuAD 2.0|12655||
|KorWIKI|10000||
|Office|6826|1397|
|Spec|2014|300|
|Law|1546|300|
|Hub 행정|-|?|

## 실험환경
batch_size = 4, epoch = 데이터별 2 ~ 11 epoch, learning_rate = 3e-5

## 실험결과
No Discriminator - Base
|Dataset|F1|EM|
|----|----|----|
|KorQuAD 2.0|86.55|78.73|
|KorWIKI|-|-|
|Office|78.00|65.46|
|Spec|41.32|32.50|
|Law|63.57|48.92|
|Hub 행정|-|-|

Discriminator
|Dataset|F1|EM|
|----|----|----|
|KorQuAD 2.0|86.37|80.65|
|KorWIKI|93.66|89.11|
|Office|81.66|71.79|
|Spec|86.10|78.33|
|Law|73.91|38.66|
|Hub 행정|90.42|85.11|
