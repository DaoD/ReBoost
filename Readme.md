# ReBoost: A Retrieval-Boosted Sequence-to-Sequence Model for Neural Response Generation
This repository contains the source code and datasets for the IR Journal paper [ReBoost: A Retrieval-Boosted Sequence-to-Sequence Model for Neural Response Generation](https://link.springer.com/content/pdf/10.1007%2Fs10791-019-09364-x.pdf) by Zhu et al. <br>

## Model overview
<img src="image/model.png">

## Results
<img src="image/automatic_results.png">
<img src="image/human_results.png">

## Dependencies
Python 3.5 <br>
Tensorflow 1.2.0

## Datasets
Your can download the processed datasets used in our paper here and unzip it to the folder of ```data```. <br>
[Weibo (STC)](https://drive.google.com/file/d/15PY2VmiqRKr-0-crwGJsrBKS5xTqubGu/view?usp=sharing) <br>

## Train and test a model
python3 runModel.py

## Cite
If you use the code and datasets, please cite the following paper:  
**"ReBoost: A Retrieval-Boosted Sequence-to-Sequence Model for Neural Response Generation"**  
Yutao Zhu, Zhicheng Dou, Jian-Yun Nie and Ji-Rong Wen. _IR Journal (2019)_

```
@article{DBLP:journals/ir/ZhuDNW20,
  author    = {Yutao Zhu and
               Zhicheng Dou and
               Jian{-}Yun Nie and
               Ji{-}Rong Wen},
  title     = {ReBoost: a retrieval-boosted sequence-to-sequence model for neural
               response generation},
  journal   = {Inf. Retr. Journal},
  volume    = {23},
  number    = {1},
  pages     = {27--48},
  year      = {2020},
  url       = {https://doi.org/10.1007/s10791-019-09364-x},
  doi       = {10.1007/s10791-019-09364-x}
}
```
