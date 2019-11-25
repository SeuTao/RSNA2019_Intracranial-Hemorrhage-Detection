import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch
import pretrainedmodels
import math
from efficientnet_pytorch import EfficientNet

class se_resnext101_32x4d(nn.Module):

    def __init__(self):
        super(se_resnext101_32x4d, self).__init__()

        self.model_ft = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, 6, bias=True))

    def forward(self, x):
        x = self.model_ft(x)
        return x

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x): 
        return torch.cat([self.ap(x), self.mp(x)], 1)

def l2_norm(input, axis=1):
    norm = torch.norm(input,2, axis, True)
    output = torch.div(input, norm)
    return output

class BinaryHead(nn.Module):

    def __init__(self, num_class, emb_size = 2048, s = 16.0):
        super(BinaryHead,self).__init__()
        self.s = s
        self.fc = nn.Sequential(nn.Linear(emb_size, num_class))

    def forward(self, fea):
        fea = l2_norm(fea)
        logit = self.fc(fea)*self.s
        return logit

class se_resnext50_32x4d(nn.Module):

    def __init__(self):
        super(se_resnext50_32x4d, self).__init__()

        self.model_ft = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, 6, bias=True))

    def forward(self, x):
        x = self.model_ft(x)
        return x

class se_resnext50_32x4d_v2(nn.Module):

    def __init__(self, classCount):
        super(se_resnext50_32x4d_v2, self).__init__()

        self.model_ft = nn.Sequential(*list(pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet').children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.last_linear = None
        self.fea_bn = nn.BatchNorm1d(2048)
        self.fea_bn.bias.requires_grad_(False)
        self.binary_head = BinaryHead(classCount, emb_size=2048, s = 16)


    def forward(self, x):

        img_feature = self.model_ft(x)
        img_feature = self.avg_pool(img_feature)
        img_feature = img_feature.view(img_feature.size(0), -1)
        fea = self.fea_bn(img_feature)
        output = self.binary_head(fea)

        return output


class DenseNet201_change_avg(nn.Module):

    def __init__(self):
    
        super(DenseNet201_change_avg, self).__init__()
        
        self.densenet201 = torchvision.models.densenet201(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1920, 6)
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):

        x = self.densenet201(x)      
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        # x = self.sigmoid(x)
        
        return x

class DenseNet169_change_avg(nn.Module):

    def __init__(self):
    
        super(DenseNet169_change_avg, self).__init__()
        
        self.densenet169 = torchvision.models.densenet169(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1664, 6)
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):

        x = self.densenet169(x)      
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        
        return x


class DenseNet121_change_avg(nn.Module):

    def __init__(self):
    
        super(DenseNet121_change_avg, self).__init__()
        
        self.densenet121 = torchvision.models.densenet121(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1024, 6)
        self.sigmoid = nn.Sigmoid()   

        
    def forward(self, x):
        x = self.densenet121(x)      
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.mlp(x)
        
        return x

class DenseNet121_change_avg_3d_features(nn.Module):

    def __init__(self):
    
        super(DenseNet121_change_avg_3d_features, self).__init__()
        
        self.densenet121 = torchvision.models.densenet121(pretrained=True).features

        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.mlp = nn.Sequential(nn.Linear(1024+512, 1024), nn.Linear(1024, 6))
        self.sigmoid = nn.Sigmoid()   

        
    def forward(self, x, y):

        x = self.densenet121(x)      
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = torch.cat((x, y), 1)
        x = self.mlp(x)
        # x = self.sigmoid(x)
        
        return x

