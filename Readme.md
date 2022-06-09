# EAKT: Embedding Cognitive Framework with Attention for Interpretable Knowledge Tracing
![Python 3.7](https://img.shields.io/badge/python-3.7-green)
![PyTorch 1.3](https://img.shields.io/badge/pytorch-1.3-orange)
![cuDNN 7.6.1](https://img.shields.io/badge/cudnn-7.6.3-blue)

<p align="center">
<img src=".\img\architecture.png" height = "400" alt="" align=center />
<br><br>
<b>Figure 1.</b> The architecture of EAKT.
</p>
# About

This is an implementation of the EAKT model, described in the following paper: EAKT: Embedding Cognitive Framework with Attention for Interpretable Knowledge Tracing.

# Contributors

- **Yanjun Pu**: yliang@buaa.edu.cn 
- **Fang Liu** : liufangg@buaa.edu.cn
- **Tianhao Peng**: pengtianhao@buaa.edu.cn


_State Key Laboratory of Software Development Environment Admire Group, School of Computer Science and Engineering, Beihang University_

# Dataset
The `csv` folder has four datasets, [assist2009](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010), [assist2015](ASSISTment2015:https://sites.google.com/site/assistmentsdata/home/2015-assistments-skill-builder-data), [assist2017](https://sites.google.com/view/assistmentsdatamining/dataset), simulation. Every dataset include three files:train data, test data, and Q-matrix data.

The statistical information of the above datasets is given in following Table.


| **Dataset**    | **Student** | **Question** | **Interaction** | **Average length** | **Maximum length** |
|----------------|-------------|--------------|-----------------|--------------------|--------------------|
| **ASSIST2009** | 4,151       | 110          | 325,637         | 78                 | 1,261              |
| **ASSIST2015** | 19,917      | 100          | 708,631         | 35                 | 632                |
| **ASSIST2017** | 1,709       | 102          | 942,816         | 551                | 3,057              |
| **Simu**       | 20,000      | 30           | 1,000,000       | 50                 | 50                 |


## Usage
```
Usage:
    run.py (eakt) --data=<h> --questions=<h> [options]

Options:
    --length=<int>                      max length of question sequence [default: 50]
    --questions=<int>                   num of question [default: 124]
    --lr=<float>                        learning rate [default: 0.001]
    --bs=<int>                          batch size [default: 64]
    --seed=<int>                        random seed [default: 59]
    --epochs=<int>                      number of epochs [default: 30]
    --cuda=<int>                        use GPU id [default: 0]
    --hidden=<int>                      dimention of hidden state [default: 128]
    --kc=<int>                          knowledge compoments dimention [default: 10]
    --layers=<int>                      layers of rnn or transformer [default: 1]
    --heads=<int>                       head number of transformer [default: 8]
    --dropout=<float>                   dropout rate [default: 0.1]
    --beta=<float>                      reduce rate of MyModel [default: 0.95]
    --data=<string>                     dataset [default: assist2009]
    --kernels=<int>                     the kernel size of CNN [default: 7]
    --memory_size=<int>                 memory size of DKVMN model [default: 20]
    --weight_decay=<float>              weight_decay of optimizer [default: 0]
    --sigmoida=<float>                  coefficient of custom sigmoid function [default: 5]
    --sigmoidb=<float>                  constant custom sigmoid function [default: 6.9]
    --save_model=<bool>                 whether save the KT model [default: true]
    --save_epoch=<int>                  the epoch to save the KT model [default: 0]
    --gpu=<int>                         select the gpu card [default: 0]
```

#### example
```
### Run EAKT model with assist2009 dataset.
`python -m evaluation.run eakt --data=assist2009updated --questions=110 --kc=30 --heads=8 --hidden=128 --bs=5 --epochs=20 --save_epoch=19 --lr=0.001 --weight_decay=0.000001 --sigmoida=1 --sigmoidb=0`

```


