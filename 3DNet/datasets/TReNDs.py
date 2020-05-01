import os
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold,StratifiedKFold, GroupKFold, KFold
import nilearn as nl
import torch
import random
from tqdm import tqdm

import monai
from monai.transforms import \
    LoadNifti, LoadNiftid, AddChanneld, ScaleIntensityRanged, \
    Rand3DElasticd, RandAffined, \
    Spacingd, Orientationd

root = r'./competition_root'

train = pd.read_csv('{}/train_scores.csv'.format(root)).sort_values(by='Id')
loadings = pd.read_csv('{}/loading.csv'.format(root))
sample = pd.read_csv('{}/sample_submission.csv'.format(root))
reveal = pd.read_csv('{}/reveal_ID_site2.csv'.format(root))
ICN = pd.read_csv('{}/ICN_numbers.csv'.format(root))

"""
    Load and display a subject's spatial map
"""

def load_subject(filename, mask_niimg):
    """
    Load a subject saved in .mat format with the version 7.3 flag. Return the subject niimg, using a mask niimg as a template for nifti headers.
    Args:
        filename    <str>            the .mat filename for the subject data
        mask_niimg  niimg object     the mask niimg object used for nifti headers
    """
    subject_data = None
    with h5py.File(filename, 'r') as f:
        subject_data = f['SM_feature'][()]
        # print(subject_data.shape)
    # It's necessary to reorient the axes, since h5py flips axis order
    subject_data = np.moveaxis(subject_data, [0, 1, 2, 3], [3, 2, 1, 0])
    # print(subject_data.shape)
    return subject_data
    # subject_niimg = nl.image.new_img_like(mask_niimg, subject_data, affine=mask_niimg.affine, copy_header=True)
    # return subject_niimg

def read_data_sample():
    # Input data files are available in the "../input/" directory.
    # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
    mask_filename = r'{}/fMRI_mask.nii'.format(root)
    subject_filename = '{}/fMRI_train/10004.mat'.format(root)

    mask_niimg = nl.image.load_img(mask_filename)
    print("mask shape is %s" % (str(mask_niimg.shape)))

    subject_niimg = load_subject(subject_filename, mask_niimg)
    print("Image shape is %s" % (str(subject_niimg.shape)))
    num_components = subject_niimg.shape[-1]
    print("Detected {num_components} spatial maps".format(num_components=num_components))

class TReNDsDataset(Dataset):

    def __init__(self, mode='train', fold_index = 0):
        # print("Processing {} datas".format(len(self.img_list)))
        self.mode = mode
        self.fold_index = fold_index

        if self.mode=='train' or self.mode=='valid' or self.mode=='valid_tta':
            features = ('age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2')
            data = pd.merge(loadings, train, on='Id').dropna()
            id_train = list(data.Id)
            fea_train = np.asarray(data.drop(list(features), axis=1).drop('Id', axis=1))
            lbl_train = np.asarray(data[list(features)])

            self.all_samples = []
            for i in range(len(id_train)):
                id = id_train[i]
                fea = fea_train[i]
                lbl = lbl_train[i]
                filename = os.path.join('{}/fMRI_train_npy/{}.npy'.format(root, id))
                self.all_samples.append([filename, fea, lbl, str(id)])

            fold = 0
            kf = KFold(n_splits=5, shuffle=True, random_state=1337)
            for train_index, valid_index in kf.split(self.all_samples):
                if fold_index == fold:
                    self.train_index = train_index
                    self.valid_index = valid_index
                fold+=1

            if self.mode=='train':
                self.train_index = [tmp for tmp in self.train_index if os.path.exists(self.all_samples[tmp][0])]
                self.len = len(self.train_index)
                print('fold index:',fold_index)
                print('train num:', self.len)

            elif self.mode=='valid' or self.mode=='valid_tta':
                self.valid_index = [tmp for tmp in self.valid_index if os.path.exists(self.all_samples[tmp][0])]
                self.len = len(self.valid_index)
                print('fold index:',fold_index)
                print('valid num:', self.len)

        elif  self.mode=='test':
            labels_df = pd.read_csv("{}/train_scores.csv".format(root))
            labels_df["is_train"] = True

            features = ('age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2')
            data = pd.merge(loadings, labels_df, on="Id", how="left")

            id_test = list(data[data["is_train"] != True].Id)
            fea_test = np.asarray(data.drop(list(features), axis=1).drop('Id', axis=1)[data["is_train"] != True].drop("is_train", axis=1))
            lbl_test = np.asarray(data[list(features)][data["is_train"] != True])

            self.all_samples = []
            for i in range(len(id_test)):
                id = id_test[i]
                fea = fea_test[i]
                lbl = lbl_test[i]

                filename = os.path.join('{}/fMRI_test_npy/{}.npy'.format(root, id))
                if os.path.exists(filename):
                    self.all_samples.append([id, filename, fea, lbl])

            self.len = len(self.all_samples)
            print(len(id_test))
            print('test num:', self.len)

    def __getitem__(self, idx):

        if self.mode == "train" :
            filename, _, lbl, id =  self.all_samples[self.train_index[idx]]
            train_img = np.load(filename).astype(np.float32)
            train_img = train_img.transpose((3,2,1,0))
            # (53, 52, 63, 53)
            train_lbl = lbl

            data_dict = {'image':train_img}
            rand_affine = RandAffined(keys=['image'],
                                      mode=('bilinear', 'nearest'),
                                      prob=0.5,
                                      spatial_size=(52, 63, 53),
                                      translate_range=(5, 5, 5),
                                      rotate_range=(np.pi * 4, np.pi * 4, np.pi * 4),
                                      scale_range=(0.15, 0.15, 0.15),
                                      padding_mode='border')
            affined_data_dict = rand_affine(data_dict)
            train_img = affined_data_dict['image']

            return torch.FloatTensor(train_img), \
                   torch.FloatTensor(train_lbl)

        elif self.mode == "valid":
            filename, _, lbl, id =  self.all_samples[self.valid_index[idx]]
            train_img = np.load(filename).astype(np.float32)
            train_img = train_img.transpose((3, 2, 1, 0))
            # (53, 52, 63, 53)
            train_lbl = lbl

            return torch.FloatTensor(train_img),\
                   torch.FloatTensor(train_lbl)

        elif self.mode == 'test':
            id, filename, fea, lbl =  self.all_samples[idx]
            test_img = np.load(filename).astype(np.float32)
            test_img = test_img.transpose((3, 2, 1, 0))

            return str(id), \
                   torch.FloatTensor(test_img)

    def __len__(self):
        return self.len

def run_check_datasets():
    dataset = TReNDsDataset(mode='test')
    for m in range(len(dataset)):
        tmp = dataset[m]
        print(m)

def convert_mat2nii2npy():

    def get_data(filename):
        with h5py.File(filename, 'r') as f:
            subject_data = f['SM_feature'][()]
            # print(subject_data.shape)
        # It's necessary to reorient the axes, since h5py flips axis order
        subject_data = np.moveaxis(subject_data, [0, 1, 2, 3], [3, 2, 1, 0])
        return subject_data

    # train_root = '{}/fMRI_train/'.format(root)
    # train_npy_root = '{}/fMRI_train_npy/'.format(root)
    train_root = '{}/fMRI_test/'.format(root)
    train_npy_root = '{}/fMRI_test_npy/'.format(root)
    os.makedirs(train_npy_root, exist_ok=True)

    mats = os.listdir(train_root)
    mats = [mat for mat in mats if '.mat' in mat]
    random.shuffle(mats)

    for mat in tqdm(mats):
        mat_path = os.path.join(train_root, mat)
        if os.path.exists(mat_path):
            print(mat_path)

        npy_path = os.path.join(train_npy_root, mat.replace('.mat','.npy'))
        if os.path.exists(npy_path):
            print(npy_path, 'exist')
        else:
            data = get_data(mat_path)
            print(npy_path,data.shape)
            np.save(npy_path,data.astype(np.float16))

if __name__ == '__main__':
    run_check_datasets()
    # convert_mat2nii2npy()




