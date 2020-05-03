
# 3DConvNet Solutions for Medical Image Challenges
This repository contains 3d ConvNet Solutions for Medical Image Challenges.
The project is based on  [Tencent MedicalNet](https://github.com/Tencent/MedicalNet) and [MONAI framework](https://monai.io/)
which provides a series of 3D-ResNet pre-trained models and domain-optimized foundational capabilities for developing healthcare imaging training workflows.

### Update(2020/05/01)
I provide a baseline 3DConvNet code for TReNDS Neuroimaging challenge host on Kaggle. 

### Contents
1. [Requirements](#Requirements)
2. [Code](#Demo)
4. [Experiments](#Experiments)
5. [TODO](#TODO)

### Requirements
- Python 3.7.0
- PyTorch-1.5
- monai-0.1.0  

### Code
- Structure of data directories base on MedicalNet
```
/
    |--datasets/：Data preprocessing module
    |   |--brains18.py：MRBrainS18 data preprocessing script
    |   |--RSNA19.py：RSNA19 data preprocessing script
    |   |--TReNDs.py：TReNDs data preprocessing script
    |--models/：Model construction module
    |   |--resnet.py：3D-ResNet network build script
    |--utils/：tools
    |   |--logger.py：Logging script
    |--toy_data/：For CI test
    |--data/：Data storage module
    |   |--MRBrainS18/：MRBrainS18 dataset
    |	|   |--images/：source image named with patient ID
    |	|   |--labels/：mask named with patient ID
    |   |--train.txt: training data lists
    |   |--val.txt: validation data lists
    |--pretrain/：Pre-trained models storage module
    |--model.py: Network processing script
    |--setting.py: Parameter setting script
    |--train_MRBrainS18.py: MRBrainS18 training demo script
    |--train_TReNDs.py: TReNDs training script
    |--train_RSNA19.py
    |--README.md
```

 Download data & pre-trained models from Tencent MedicalNet official repo ([Google Drive](https://drive.google.com/file/d/1399AsrYpQDi1vq6ciKRQkfknLsQQyigM/view?usp=sharing) or [Tencent Weiyun](https://share.weiyun.com/55sZyIx))
 
- Network structure parameter settings
```
Model name   : parameters settings
resnet_10.pth: --model resnet --model_depth 10 --resnet_shortcut B
resnet_18.pth: --model resnet --model_depth 18 --resnet_shortcut A
resnet_34.pth: --model resnet --model_depth 34 --resnet_shortcut A
resnet_50.pth: --model resnet --model_depth 50 --resnet_shortcut B
resnet_101.pth: --model resnet --model_depth 101 --resnet_shortcut B
resnet_152.pth: --model resnet --model_depth 152 --resnet_shortcut B
resnet_200.pth: --model resnet --model_depth 200 --resnet_shortcut B
```

### Baseline for TReNDS Neuroimaging challenge
- 3D-Resnet10 trained from scratch [Pretrained models](https://drive.google.com/open?id=1mB59NoADt0n4yC-MviMtBUcYCE2YWJZz)

Network | fold 0| fold 1| fold 2| fold 3| fold 4|
---|---|---|---|---|---|
3D-Resnet10 Train from scratch|0.1700|0.1685|0.1729|0.1734|0.1734
3D-Resnet10 MedicalNet pretrained|0.1694|0.1691|0.1726|0.1746|0.1734

### Computational Cost 
```
GPU：NVIDIA Tesla P40
```
<table class="dataintable">
<tr>
   <th class="dataintable">Network</th>
   <th>Paramerers (M)</th>
   <th>Running time (s)</th>
</tr>
<tr>
   <td>3D-ResNet10</td>
   <td>14.36</td>
   <td>0.18</td>
</tr class="dataintable">
<tr>
   <td>3D-ResNet18</td>
   <td>32.99</td>
   <td>0.19</td>
</tr>
<tr>
   <td>3D-ResNet34</td>
   <td>63.31</td>
   <td>0.22</td>
</tr>
<tr>
   <td>3D-ResNet50</td>
   <td>46.21</td>
   <td>0.21</td>
</tr>
<tr>
   <td>3D-ResNet101</td>
   <td>85.31</td>
   <td>0.29</td>
</tr>
<tr>
   <td>3D-ResNet152</td>
   <td>117.51</td>
   <td>0.34</td>
</tr>
<tr>
   <td>3D-ResNet200</td>
   <td>126.74</td>
   <td>0.45</td>
</tr>
</table>

- Please refer to [Med3D: Transfer Learning for 3D Medical Image Analysis](https://arxiv.org/abs/1904.00625) for more details：

### TODO
- [x] Baseline (pure 3D ConvNet) code for TReNDS Neuroimaging challenge
- [ ] Code and pretrained models for Intracranial-Hemorrhage-Detection (RSNA2019 challenge dataset)
- [ ] More baseline code and pretrained models for recent Medical Image Challenges

