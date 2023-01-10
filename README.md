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
Model : KoBigBird Model

|Dataset|trained Data|Test Data|
|----|-----|------|
|KorQuAD 2.0|12655||
|KorWIKI|10000||
|Office|6826|1397|
|Spec|2014|300|
|Law|1546|300|

## 실험환경
batch_size = 4, epoch = 데이터별 2 ~ 11 epoch
