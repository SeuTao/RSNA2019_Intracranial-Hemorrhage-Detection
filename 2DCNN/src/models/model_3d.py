import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import itertools
import torch.nn.functional as F
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import nibabel as nib  
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR
import gc, os, csv
import matplotlib.pyplot as plt
import itertools
import numpy as np
import cv2
import torch.utils.data as data
import random
from models.medicalnet_resnet import generate_model

class medicalnet_parameter:
    def __init__(self):
        self.model = 'resnet'
        self.phase = 'train'
        self.no_cuda = True
        self.pretrain_path = '/data/VPS/VPS_04/kaggle/kaggle_rsna2019/models_snapshot/medicalnet_pretrain_model/pretrain/resnet_34_23dataset.pth'
        self.num_workers = 0
        self.model_depth = 34
        self.resnet_shortcut = 'A'
        self.input_D = 32
        self.input_H = 128
        self.input_W = 128
        self.n_seg_classes = 2
        self.new_layer_names = []

class medicalnet_resnet34(nn.Module):
    def __init__(self):
        super(medicalnet_resnet34, self).__init__()

        self.parameter = medicalnet_parameter()
        self.model, parameters = generate_model(self.parameter) 
        self.cls_head = nn.Sequential(nn.Linear(512, 512, bias=True), nn.Linear(512, 6, bias=True))

    def forward(self, x): 
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = nn.AdaptiveAvgPool3d((1,1,1))(x)
        x = x.view(x.size(0), -1)
        x = self.cls_head(x)

        return x