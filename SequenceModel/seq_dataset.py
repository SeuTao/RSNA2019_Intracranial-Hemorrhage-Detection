import os
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import torch
from check_oof import *
from check_feature import *
from settings import *
import random
import numpy as np

class StackingDataset_study(Dataset):
    def __init__(self, dict_, X,Y, index, seq_len = 32, mode='train' , reverse = False, Add_position = False):

        self.mode = mode
        self.study_dict = {}
        self.dict_ = dict_
        self.Add_position = Add_position

        self.reverse = reverse
        print(self.reverse)
        self.X = X
        self.Y = Y

        self.seq_len = seq_len
        self.mode = mode

        if mode == 'train' or mode == 'valid':
            self.all_df = pd.read_csv(rf'{csv_root}/train_meta_id_seriser.csv')
            self.StudyInstance = list(self.all_df['StudyInstance'].unique())
            self.index = index
            self.len = len(index)

        elif mode == 'test':
            self.index = index
            self.all_df = pd.read_csv(rf'{csv_root}/test_meta_id_seriser_stage2.csv')
            self.StudyInstance = list(self.all_df['StudyInstance'].unique())
            self.len = len(self.StudyInstance)

        self.study_dict = {}
        print('mode: '+self.mode)
        print(self.len)

    def __getitem__(self, index):

        if self.mode == 'train' or self.mode == 'valid':
            index = self.index[index]
            StudyInstance = self.StudyInstance[index]
        else:
            StudyInstance = self.StudyInstance[index]

        if StudyInstance not in self.study_dict:
            self.study_dict[StudyInstance] = pd.read_csv(os.path.join(csv_root, 'study_csv', StudyInstance + '.csv'))

        same_StudyInstance = self.study_dict[StudyInstance]
        same_StudyInstance.reset_index(drop=True)

        all_index = same_StudyInstance.index.values#.to_list()
        Position2 = same_StudyInstance.Position2.values#.to_list()

        if self.mode == 'train' and len(all_index) > 10:
            if random.randint(0,1) == 0:
                rd = random.randint(0, 1)
                all_index = [all_index[i] for i in range(len(all_index)) if i %2 == rd]
                Position2 = [Position2[i] for i in range(len(Position2)) if i %2 == rd]

        if self.Add_position:
            Position2 = [Position2[i + 1] - Position2[i] for i in range(len(Position2) - 1)]
            Position2.append(Position2[-1])
            X_position2 = np.asarray(Position2).reshape([-1, 1, 1])
            X_position = np.concatenate([X_position2, X_position2, X_position2,X_position2, X_position2, X_position2, ], axis=1)

        if self.mode == 'train' or self.mode == 'valid':
            X_tmp = [self.X[self.dict_[same_StudyInstance.iloc[tmp, 1].replace('.dcm','')]] for tmp in all_index]
            Y_tmp = [self.Y[self.dict_[same_StudyInstance.iloc[tmp, 1].replace('.dcm','')]] for tmp in all_index]

            fea_tmp = [train_fea[fea_id_dict[same_StudyInstance.iloc[tmp, 1].replace('.dcm','')]] for tmp in all_index]
            fea_tmp = np.asarray(fea_tmp)

            X_tmp = np.asarray(X_tmp)
            Y_tmp = np.asarray(Y_tmp)

            if self.seq_len > 0:
                s = 0
                if X_tmp.shape[0] > self.seq_len:
                    s = random.randint(0, X_tmp.shape[0]-self.seq_len)

                X_tmp = X_tmp[s:s+self.seq_len, :, :]
                Y_tmp = Y_tmp[s:s+self.seq_len, :]
                fea_tmp = fea_tmp[s:s+self.seq_len, :,:]
                fea = np.zeros([self.seq_len, feature_dim, feature_num])

                if self.Add_position:
                    X_position_tmp = X_position[s:s+self.seq_len, :, :]
                    X_tmp = np.concatenate([X_tmp, X_position_tmp],axis=2)

                    X = np.zeros([self.seq_len, 6, model_num+1])
                    Y = np.zeros([self.seq_len, 6])
                else:
                    X = np.zeros([self.seq_len, 6, model_num])
                    Y = np.zeros([self.seq_len, 6])

                if self.mode == 'train' and random.randint(0, 1) == 0:
                    X[0:X_tmp.shape[0], :, :] = X_tmp[::-1, : , :]
                    Y[0:Y_tmp.shape[0], :]    = Y_tmp[::-1, :]
                    fea[0:Y_tmp.shape[0], :,:]  = fea_tmp[::-1, :,:]

                elif self.mode == 'valid' and self.reverse:
                    X[0:X_tmp.shape[0], :, :] = X_tmp[::-1, :, :]
                    Y[0:Y_tmp.shape[0], :]    = Y_tmp[::-1, :]
                    fea[0:Y_tmp.shape[0], :,:]  = fea_tmp[::-1, :,:]
                else:
                    X[0:X_tmp.shape[0], :, :] = X_tmp
                    Y[0:Y_tmp.shape[0], :] = Y_tmp
                    fea[0:Y_tmp.shape[0], :,:]  = fea_tmp

            else:
                X_tmp = np.asarray(X_tmp)

                if self.Add_position:
                    X_position_tmp = X_position
                    X_tmp = np.concatenate([X_tmp, X_position_tmp], axis=2)

                if self.reverse:
                    shape = np.asarray(X_tmp).shape
                    X = np.zeros(shape)
                    shape = np.asarray(Y_tmp).shape
                    Y = np.zeros(shape)

                    X[:,:,:] = np.asarray(X_tmp)[::-1, :, :]
                    Y[:,:]   = np.asarray(Y_tmp)[::-1, :]

                    shape = fea_tmp.shape
                    fea = np.zeros(shape)
                    fea[:,:,:] = fea_tmp[::-1, :,:]
                else:
                    X = np.asarray(X_tmp)
                    Y = np.asarray(Y_tmp)
                    fea = fea_tmp

            X = X.transpose(2,0,1)
            X = torch.from_numpy(X)
            Y = torch.from_numpy(Y)

            # self.seq_len, feature_dim, feature_num
            fea = fea.reshape([-1, feature_dim, feature_num])
            fea = fea.transpose(1,0,2)
            fea = torch.from_numpy(fea)
            return fea, X, Y

        elif self.mode == 'test':
            X_tmp = [self.X[self.dict_[same_StudyInstance.iloc[tmp, 1].replace('.dcm', '')]] for tmp in all_index]
            fea_tmp = [test_fea[fea_id_dict[same_StudyInstance.iloc[tmp, 1].replace('.dcm', '')]] for tmp in all_index]
            fea_tmp = np.asarray(fea_tmp)

            if self.Add_position:
                X_position_tmp = X_position
                X_tmp = np.concatenate([X_tmp, X_position_tmp], axis=2)

            if self.reverse:
                shape = X_tmp.shape
                X = np.zeros(shape)
                X[:, :, :] = X_tmp[::-1, :, :]

                shape = fea_tmp.shape
                fea = np.zeros(shape)
                fea[:, :,:] = fea_tmp[::-1, :,:]
            else:
                X = np.asarray(X_tmp)
                fea = fea_tmp

            X = X.transpose(2, 0, 1)
            X = torch.from_numpy(X)

            fea = fea.reshape([-1, feature_dim, feature_num])
            fea = fea.transpose(1, 0, 2)
            fea = torch.from_numpy(fea)

            filenames = [same_StudyInstance.iloc[tmp, 1].replace('.dcm', '') for tmp in all_index]
            return filenames, fea, X

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

def run_check_train_data():
    kf = KFold(n_splits=5, shuffle=True, random_state=48)
    all_df = pd.read_csv(rf'{csv_root}/train_meta_id_seriser.csv')
    StudyInstance = list(all_df['StudyInstance'].unique())
    print(len(StudyInstance))
    dict_ = get_train_dict()
    for s_fold, (train_idx, val_idx) in enumerate(kf.split(StudyInstance)):
        dataset = StackingDataset_study(dict_, X,y, train_idx, seq_len = 32, mode='valid', reverse=True, Add_position=True)
        num = len(dataset)
        for m in range(num):
            i = np.random.choice(num)
            fea, image, label= dataset[i]
            print(fea.shape)
            print(image.shape)
            print(label.shape)

def run_check_test_data():
    test_id_dict = get_test_dict()
    dataset = StackingDataset_study(test_id_dict, X_test, None, None, seq_len=-1, mode='test')

    num = len(dataset)
    for m in range(num):
        i = np.random.choice(num)
        ids, image = dataset[i]
        print(ids)
        print(image.shape)

if __name__ == '__main__':
    run_check_train_data()
    run_check_test_data()



