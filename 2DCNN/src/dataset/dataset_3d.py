from torchvision import transforms
import torchvision.datasets as datasets
from PIL import Image
import torch.utils.data as data
import torch
import numpy as np
import cv2
from tqdm import tqdm
import random
import albumentations
from torch.utils.data.dataloader import default_collate
import torch.utils.data.sampler as torchSampler
import pandas as pd
from torch.utils.data.sampler import Sampler
from scipy import ndimage

def crop_image(image):

    z_shape = image.shape[0]
    middle = z_shape//2
    if z_shape == 32:
        new_image = image
    elif z_shape > 32:
        new_image = image[middle-16:middle+16,:,:]
    else:
        if z_shape%2 == 0:
            new_image = np.zeros((32,image.shape[1], image.shape[2]))
            new_image[16-middle:16+middle,:,:] = image
        else:
            new_image = np.zeros((32,image.shape[1], image.shape[2]))
            new_image[16-middle-1:16+middle,:,:] = image
    return new_image

def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """

    pixels = volume[volume > 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/std
    out_random = np.random.normal(0, 1, size = volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out

def resize_data(data):
    """
    Resize the data to the input size
    """ 
    [depth, height, width] = data.shape
    scale = [1.0, 128*1.0/height, 128*1.0/width]  
    data = ndimage.interpolation.zoom(data, scale, order=0)
    return data

class RSNA_Dataset_train_3d(data.Dataset):
    def __init__(self,
                 df = None,
                 name_list = None
                 ):
        self.df = pd.read_csv("/data/raw_data_repository/kaggle/kaggle_rsna2019/stage1_train_cls.csv")
        self.train_df_study = self.df.groupby('study_instance_uid').sum()
        self.train_df_study['study_instance_uid'] = self.train_df_study.index
        self.train_df_study = self.train_df_study[self.train_df_study['study_instance_uid'].isin(name_list)]
        self.name_list = name_list

    def __getitem__(self, idx):
        
        name = self.name_list[idx % len(self.name_list)]
        image = np.load('/home1/kaggle_rsna2019/process/train_study_256/' + name + '.npy')
        image = crop_image(image)
        # image = resize_data(image)
        label = torch.FloatTensor(self.train_df_study[self.train_df_study['study_instance_uid']==name].loc[:, 'any':'subdural'].values)
        label[label>0.5]=1
        image = itensity_normalize_one_volume(image)
        image = np.expand_dims(image,axis=0)

        return image, label


    def __len__(self):
        return len(self.name_list)


class RSNA_Dataset_val_3d(data.Dataset):
    def __init__(self,
                 df = None,
                 name_list = None
                 ):
        self.df = pd.read_csv("/data/raw_data_repository/kaggle/kaggle_rsna2019/stage1_train_cls.csv")
        self.train_df_study = self.df.groupby('study_instance_uid').sum()
        self.train_df_study['study_instance_uid'] = self.train_df_study.index
        self.train_df_study = self.train_df_study[self.train_df_study['study_instance_uid'].isin(name_list)]
        self.name_list = name_list

    def __getitem__(self, idx):
        
        name = self.name_list[idx % len(self.name_list)]
        image = np.load('/home1/kaggle_rsna2019/process/train_study_256/' + name + '.npy')
        image = crop_image(image)
        # image = resize_data(image)
        label = torch.FloatTensor(self.train_df_study[self.train_df_study['study_instance_uid']==name].loc[:, 'any':'subdural'].values)
        label[label>0.5]=1
        image = itensity_normalize_one_volume(image)
        image = np.expand_dims(image,axis=0)

        return image, label

    def __len__(self):
        return len(self.name_list)



def generate_dataset_loader_cls_seg(df_all, c_train, train_batch_size, c_val, val_batch_size, workers):

    train_dataset = RSNA_Dataset_train_3d(df_all, c_train)
    val_dataset = RSNA_Dataset_val_3d(df_all, c_val)
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,        
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,        
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False)

    return train_loader, val_loader


def read_list_from_txt(file_name):

    f_dataset = open(file_name, 'r')
    c_dataset = f_dataset.readlines()
    f_dataset.close()
    c_dataset = [s.replace('\n', '') for s in c_dataset]

    return c_dataset
