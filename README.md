# Automotive cyber threat intelligence vith SSMs
A derivative from original repository [here](https://github.com/AutoCS-wyh/Automotive-cyber-threat-intelligence-corpus).
This repository includes the Torch impelemtation of the original paddlepaddle solutions and incorporating the SSMs into the original architectures.

## Data description
Raw data: unstructured cybersecurity data (.txt files)

Brat annotation data：annotation data files using brat tool (.ann files) 

BIOES："BIOES" - "entity type" - "relation type" - "entity role" joint annotation data (.txt files)

## Source code description
In order to successfuly recreate the results please:
1. Run [requirements.txt](./requirements.txt) in a Python 3.7 or newer. In case of CPU usage and will to run the original code, please uncomment the _paddlepaddle_ library and comment the _paddlepaddle-gpu_ - the conflict will be generated if both of them will be installed together. Switching between them is necessery in accordance to used processing unit.
2. Format conversion: [BIOES joint annotation.py](./BIOES%20joint%20annotation.py)
3. Perform preprocessing: [read.py](./read.py)
4. Perform scenario-specific preprocessing: [BERT-BiLSTM-att-CRF](./model/BERT-BiLSTM-att-CRF/script/preprocess.py); [BiLSTM-dynamic-att-LSTM](./model/BiLSTM-dynamic-att-LSTM/script/preprocess.py)
5. Original deep learning model training: [BERT-BiLSTM-att-CRF](./model/BERT-BiLSTM-att-CRF/script/bert+bilstm+attention+crf.py); [BiLSTM-dynamic-att-LSTM](./model/BiLSTM-dynamic-att-LSTM/script/BiLSTM-dynamic-att-LSTM.py)
6. Derivative deep learning model training: [BERT-BiLSTM-att-CRF with S4 and mamba](./model/BERT-BiLSTM-att-CRF/script/bert+bilstm+attention+crf-Torch.py); [BiLSTM-dynamic-att-LSTM with S4 and mamba](./model/BiLSTM-dynamic-att-LSTM/script/BiLSTM-dynamic-att-LSTM-Torch.py)

_Please be aware that to run an **6.**, you will be prompted to log in to your wandb account for training statistics gathering_

## Brat tool

https://github.com/nlplab/brat/archive/refs/tags/v1.3p1.tar.gz. 
