# PAN-pytorch
A Pytorch implementation of [Pyramid Attention Networks for Semantic Segmentation](https://arxiv.org/abs/1805.10180) from 2018 paper by Hanchao Li, Pengfei Xiong, Jie An, Lingxue Wang.
![image](https://github.com/JaveyWang/PAN-pytorch/blob/master/results/architecture.png)

# Installation
* Env: Python3.6, [Pytorch1.0-preview](https://pytorch.org/)
* Clone this repository.
* Download the dataset by following the instructions below.

# VOC2012 Dataset
The overall dataset is augmented by Semantic Boundaries Dataset, resulting in training data 10582 and test data 1449. https://www.sun11.me/blog/2018/how-to-use-10582-trainaug-images-on-DeeplabV3-code/

After preparing the data, please change the directory below for training.
```python
training_data = Voc2012('/home/tom/DISK/DISK2/jian/PASCAL/VOC2012', 'train_aug', transform=train_transforms)
test_data = Voc2012('/home/tom/DISK/DISK2/jian/PASCAL/VOC2012', 'val',transform=test_transforms)
```

# Evaluation
![image](https://github.com/JaveyWang/PAN-pytorch/blob/master/results/result.png)

Pixel acc|mIOU
:---------:|:----:
93.19% |78.498%
