#============ Basic imports ============#e
import os
import sys
sys.path.insert(0, '..')
import time
import gc
import pandas as pd
import cv2
import csv
from torch.utils.data import DataLoader
from dataset.dataset import *
from tuils.tools import *
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import random
import math
import argparse

def randomHorizontalFlip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
    return image

def randomVerticleFlip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
    return image

def randomRotate90(image, u=0.5):
    if np.random.random() < u:
        image[:,:,0:3] = np.rot90(image[:,:,0:3])
    return image

def random_cropping(image, ratio=0.8, is_random = True):
    height, width, _ = image.shape
    target_h = int(height*ratio)
    target_w = int(width*ratio)

    if is_random:
        start_x = random.randint(0, width - target_w)
        start_y = random.randint(0, height - target_h)
    else:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2

    zeros = image[start_y:start_y+target_h,start_x:start_x+target_w,:]
    zeros = cv2.resize(zeros ,(width,height))
    return zeros

def cropping(image, ratio=0.8, code = 0):
    height, width, _ = image.shape
    target_h = int(height*ratio)
    target_w = int(width*ratio)

    if code==0:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2

    elif code == 1:
        start_x = 0
        start_y = 0

    elif code == 2:
        start_x = width - target_w
        start_y = 0

    elif code == 3:
        start_x = 0
        start_y = height - target_h

    elif code == 4:
        start_x = width - target_w
        start_y = height - target_h

    elif code == -1:
        return image

    zeros = image[start_y:start_y+target_h,start_x:start_x+target_w,:]
    zeros = cv2.resize(zeros ,(width,height))
    return zeros

def random_erasing(img, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
    if random.uniform(0, 1) > probability:
        return img

    for attempt in range(100):
        area = img.shape[0] * img.shape[1]

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)
            if img.shape[2] == 3:
                img[x1:x1 + h, y1:y1 + w, :] = 0.0
            else:
                print('!!!!!!!! random_erasing dim wrong!!!!!!!!!!!')
                return

            return img

    return img

def randomShiftScaleRotate(image,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):

    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
    return image


def aug_image(image, is_infer=False, augment = [0,0]):

    if is_infer:
        image = randomHorizontalFlip(image, u=augment[0])
        image = np.asarray(image)
        image = cropping(image, ratio=0.8, code=augment[1])
        return image

    else:
        image = randomHorizontalFlip(image)
        height, width, _ = image.shape
        image = randomShiftScaleRotate(image,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-30, 30))

        image = cv2.resize(image, (width, height))
        image = random_erasing(image, probability=0.5, sl=0.02, sh=0.4, r1=0.3)

        ratio = random.uniform(0.6,0.99)
        image = random_cropping(image, ratio=ratio, is_random=True)

        return image

valid_transform_aug = albumentations.Compose([

    albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0, p=1.0)
])

valid_transform_pure = albumentations.Compose([

    albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0, p=1.0)
])

class PredictionDatasetPure:
    def __init__(self, name_list, df_train, df_test, n_test_aug, mode):
        self.name_list = name_list
        if mode == 'val':
            self.df = df_train[df_train['filename'].isin(name_list)]
        elif mode == 'test':
            self.df = df_test[df_test['filename'].isin(name_list)]
        self.n_test_aug = n_test_aug
        self.mode = mode

    def __len__(self):
        return len(self.name_list) * self.n_test_aug

    def __getitem__(self, idx):
        if self.mode == 'val':
            filename = self.name_list[idx % len(self.name_list)]
            image_cat = cv2.imread('/home1/kaggle_rsna2019/process/train_concat_3images_256/' + filename)
            label = torch.FloatTensor(self.df[self.df['filename']==filename].loc[:, 'any':'subdural'].values)

        if self.mode == 'test':
            filename = self.name_list[idx % len(self.name_list)]
            image_cat = cv2.imread('/home1/kaggle_rsna2019/process/stage2_test_concat_3images/' + filename)
            image_cat = cv2.resize(image_cat, (256, 256))
            label = torch.FloatTensor([0,0,0,0,0,0])

        image_cat = aug_image(image_cat, is_infer=True)
        image_cat = valid_transform_pure(image=image_cat)['image'].transpose(2, 0, 1)
        
        return filename, image_cat, label

class PredictionDatasetAug:
    def __init__(self, name_list, df_train, df_test, n_test_aug, mode):
        self.name_list = name_list
        if mode == 'val':
            self.df = df_train[df_train['filename'].isin(name_list)]
        elif mode == 'test':
            self.df = df_test[df_test['filename'].isin(name_list)]
        self.n_test_aug = n_test_aug
        self.mode = mode

    def __len__(self):
        return len(self.name_list) * self.n_test_aug

    def __getitem__(self, idx):
        if self.mode == 'val':
            filename = self.name_list[idx % len(self.name_list)]
            image_cat = cv2.imread('/home1/kaggle_rsna2019/process/train_concat_3images_256/' + filename)
            image_cat = cv2.resize(image_cat, (256, 256))
            label = torch.FloatTensor(self.df[self.df['filename']==filename].loc[:, 'any':'subdural'].values)
        if self.mode == 'test':
            filename = self.name_list[idx % len(self.name_list)]
            image_cat = cv2.imread('/home1/kaggle_rsna2019/process/stage2_test_concat_3images/' + filename)
            image_cat = cv2.resize(image_cat, (256, 256))
            label = torch.FloatTensor([0,0,0,0,0,0])

        if random.random() < 0.5:
            image_cat = cv2.cvtColor(image_cat, cv2.COLOR_BGR2RGB)
        else:
            image_cat

        image_cat = randomHorizontalFlip(image_cat, u=0.5)
        height, width, _ = image_cat.shape
        ratio = random.uniform(0.6,0.99)
        image_cat = random_cropping(image_cat, ratio=ratio, is_random=True)
        image_cat = valid_transform_aug(image=image_cat)['image'].transpose(2, 0, 1)
        
        return filename, image_cat, label

def predict(model, name_list, df_all, df_test, batch_size: int, n_test_aug: int, aug=False, mode='val', fold=0):
    if aug:
        loader = DataLoader(
            dataset=PredictionDatasetAug(name_list, df_all, df_test, n_test_aug, mode),
            shuffle=False,
            batch_size=batch_size,
            num_workers=16,
            pin_memory=True
        )
    else:
        loader = DataLoader(
            dataset=PredictionDatasetPure(name_list, df_all, df_test, n_test_aug, mode),
            shuffle=False,
            batch_size=batch_size,
            num_workers=16,
            pin_memory=True
        )

    model.eval()

    all_names = []
    all_outputs = torch.FloatTensor().cuda()
    all_truth = torch.FloatTensor().cuda()

    features_list = {}
    for names, inputs, labels in tqdm(loader, desc='Predict'):
        labels = labels.view(-1, 6).contiguous().cuda(async=True)
        all_truth = torch.cat((all_truth, labels), 0)
        with torch.no_grad():
            inputs = torch.autograd.variable(inputs).cuda(async=True)

        if backbone == 'DenseNet121_change_avg':
            feature = model.module.densenet121(inputs)      
            feature = model.module.relu(feature)
            feature = model.module.avgpool(feature)
            feature = feature.view(feature.size(0), -1)
            for index, name in enumerate(names):
                if name not in features_list:
                    features_list[name] = feature[index,:].cpu().detach().numpy()/10
                else:
                    features_list[name] += feature[index,:].cpu().detach().numpy()/10
            feature = model.module.mlp(feature)

        elif backbone == 'DenseNet169_change_avg':
            feature = model.module.densenet169(inputs)      
            feature = model.module.relu(feature)
            feature = model.module.avgpool(feature)
            feature = feature.view(feature.size(0), -1)
            for index, name in enumerate(names):
                if name not in features_list:
                    features_list[name] = feature[index,:].cpu().detach().numpy()/10
                else:
                    features_list[name] += feature[index,:].cpu().detach().numpy()/10
            feature = model.module.mlp(feature)

        elif backbone == 'se_resnext101_32x4d':
            feature = model.module.model_ft.layer0(inputs)
            feature = model.module.model_ft.layer1(feature)
            feature = model.module.model_ft.layer2(feature)
            feature = model.module.model_ft.layer3(feature)
            feature = model.module.model_ft.layer4(feature)
            feature = model.module.model_ft.avg_pool(feature)

            feature = feature.view(feature.size(0), -1)

            for index, name in enumerate(names):
                if name not in features_list:
                    features_list[name] = feature[index,:].cpu().detach().numpy()/10
                else:
                    features_list[name] += feature[index,:].cpu().detach().numpy()/10

            feature = model.module.model_ft.last_linear(feature)

            
        feature = feature.sigmoid()
        all_outputs = torch.cat((all_outputs, feature.data), 0)
        all_names.extend(names)

    for key in features_list.keys():
        if mode == 'val':
            np.save(model_snapshot_path + 'prediction/npy_train/' + key.replace('.png', '.npy'), features_list[key].astype(np.float16))
        else:
            np.save(model_snapshot_path + 'prediction/npy_test/' + key.replace('.png', '_'+str(fold)+'.npy'), features_list[key].astype(np.float16))

    datanpGT = all_truth.cpu().numpy()
    datanpPRED = all_outputs.cpu().numpy()
    return datanpPRED, all_names, datanpGT

def group_aug(val_p_aug, val_names_aug, val_truth_aug):
    """
    Average augmented predictions
    :param val_p_aug:
    :return:
    """
    df_prob = pd.DataFrame(val_p_aug)
    df_prob['id'] = val_names_aug
    
    df_truth = pd.DataFrame(val_truth_aug)
    df_truth['id'] = val_names_aug

    g_prob = df_prob.groupby('id').mean()
    g_prob = g_prob.reset_index()
    g_prob = g_prob.sort_values(by='id')
    
    g_truth = df_truth.groupby('id').mean()
    g_truth = g_truth.reset_index()
    g_truth = g_truth.sort_values(by='id')

    return g_prob.drop('id', 1).values, g_truth['id'].values, g_truth.drop('id', 1).values


def predict_all(model_name, image_size):

    for fold in [0,1,2,3,4]:

        print(fold)
        
        if not os.path.exists(model_snapshot_path + 'prediction/'):
            os.makedirs(model_snapshot_path + 'prediction/')
        if not os.path.exists(model_snapshot_path + 'prediction/npy_train/'):
            os.makedirs(model_snapshot_path + 'prediction/npy_train/')
        if not os.path.exists(model_snapshot_path + 'prediction/npy_test/'):
            os.makedirs(model_snapshot_path + 'prediction/npy_test/')

        prediction_path = model_snapshot_path+'prediction/fold_{fold}'.format(fold=fold)

        f_val = open(kfold_path + 'fold{fold}/val.txt'.format(fold=fold), 'r')
        c_val = f_val.readlines()
        f_val.close()
        c_val = [s.replace('\n', '') for s in c_val]
            
        model = eval(model_name+'()')
        model = nn.DataParallel(model).cuda()

        state = torch.load(model_snapshot_path + 'model_epoch_best_{fold}.pth'.format(fold=fold))

        epoch = state['epoch']
        best_valid_loss = state['valLoss']
        model.load_state_dict(state['state_dict'])
        print(epoch, best_valid_loss)

        model.eval()
        
        if is_center:
            val_p, val_names, val_truth = predict(model, c_val, df_all, df_test, batch_size, 1, False, 'val', fold)
            val_predictions, val_image_names, val_truth = group_aug(val_p, val_names, val_truth)
            val_loss, val_loss_sum = weighted_log_loss_numpy(val_predictions, val_truth)
            print('val_loss = ', val_loss, 'val_loss_sum = ', val_loss_sum)
            
            df = pd.DataFrame(data=val_predictions,  columns=['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'])
            df['filename'] = val_image_names
            df.to_csv(prediction_path + '_val_center.csv')

        if is_aug:
            val_p_aug, val_names_aug, val_truth_aug = predict(model, c_val, df_all, df_test, batch_size, num_aug, True, 'val', fold)
            val_predictions_aug, val_image_names_aug, val_truth_aug = group_aug(val_p_aug, val_names_aug, val_truth_aug)
            val_loss_aug, val_loss_sum_aug = weighted_log_loss_numpy(val_predictions_aug, val_truth_aug)
            print('val_loss_aug = ', val_loss_aug, 'val_loss_sum_aug = ', val_loss_sum_aug)
            df = pd.DataFrame(data=val_predictions_aug,  columns=['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'])
            df['filename'] = val_image_names_aug
            df.to_csv(prediction_path + '_val_aug_{num_aug}.csv'.format(num_aug=num_aug))

            test_p_aug, test_names_aug, test_truth_aug = predict(model, c_test, df_all, df_test, batch_size, num_aug, True, 'test', fold)
            test_predictions_aug, test_image_names_aug, test_truth_aug = group_aug(test_p_aug, test_names_aug, test_truth_aug)

            df = pd.DataFrame(data=test_predictions_aug,  columns=['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'])
            df['filename'] = test_image_names_aug
            df.to_csv(prediction_path + '_test_aug_{num_aug}.csv'.format(num_aug=num_aug))
            

        with open(model_snapshot_path + 'prediction/{model_name}.csv'.format(model_name=model_name), 'a', newline='') as f:
            writer = csv.writer(f)
            if is_center:
                writer.writerow([fold, 1, val_loss, val_loss_sum])  
            if is_aug:
                writer.writerow([fold, num_aug, val_loss_aug, val_loss_sum_aug])  

    val_lists_center = []
    test_lists_center = []
    val_lists_aug = []
    test_lists_aug = []

    prediction_path = model_snapshot_path + 'prediction/'
    for fold in range(5):
        if is_center:
            df_val_center = pd.read_csv(prediction_path + 'fold_{fold}_val_center.csv'.format(fold=fold), index_col=0 )
            val_lists_center.append(df_val_center)
            df_test_center = pd.read_csv(prediction_path + 'fold_{fold}_test_center.csv'.format(fold=fold), index_col=0)
            test_lists_center.append(df_test_center)

        if is_aug:
            df_val_aug = pd.read_csv(prediction_path + 'fold_{fold}_val_aug_{num_aug}.csv'.format(fold=fold, num_aug=num_aug), index_col=0)
            val_lists_aug.append(df_val_aug)
            df_test_aug = pd.read_csv(prediction_path + 'fold_{fold}_test_aug_{num_aug}.csv'.format(fold=fold, num_aug=num_aug), index_col=0)
            test_lists_aug.append(df_test_aug)

    if is_center:
        df_val_center = pd.concat(val_lists_center)
        df_val_center = df_val_center.sort_values(by='filename').reset_index(drop = True)
        df_val_center.to_csv(prediction_path + 'val_center.csv', index=0)

        val_predictions_center = df_val_center.loc[:, ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']].values
        val_loss_center, val_loss_sum_center = weighted_log_loss_numpy(val_predictions_center, val_truth_oof)
        print('center: ', val_loss_center, val_loss_sum_center)

    if is_aug:
        df_val_aug = pd.concat(val_lists_aug)
        df_val_aug = df_val_aug.sort_values(by='filename').reset_index(drop = True)
        df_val_aug.to_csv(prediction_path + 'val_aug_{num_aug}.csv'.format(num_aug=num_aug), index=0)

        val_predictions_aug = df_val_aug.loc[:, ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']].values
        val_loss_aug, val_loss_sum_aug = weighted_log_loss_numpy(val_predictions_aug, val_truth_oof)
        print('aug: ', val_loss_aug, val_loss_sum_aug)


    with open(model_snapshot_path + 'prediction/{model_name}.csv'.format(model_name=model_name), 'a', newline='') as f:
        writer = csv.writer(f)
        if is_center:
            writer.writerow(['center: ', val_loss_center, val_loss_sum_center])  
        if is_aug:
            writer.writerow(['aug: ', val_loss_aug, val_loss_sum_aug])  

    if is_center:
        df_test_center = pd.concat(test_lists_center)
        df_test_center = df_test_center.groupby('filename').mean()
        df_test_center.to_csv(prediction_path + 'test_center.csv')
        
    if is_aug:
        df_test_aug = pd.concat(test_lists_aug)
        df_test_aug = pd.concat(test_lists_aug)
        df_test_aug = df_test_aug.groupby('filename').mean()
        df_test_aug.to_csv(prediction_path + 'test_aug_{num_aug}.csv'.format(num_aug=num_aug))
    
    
if __name__ == '__main__':
    csv_path = '../data/stage1_train_cls.csv'
    test_csv_path = '../data/stage2_test_cls.csv'

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-backbone", "--backbone", type=str, default='DenseNet121_change_avg', help='backbone')
    parser.add_argument("-img_size", "--Image_size", type=int, default=1024, help='image_size')
    parser.add_argument("-tbs", "--train_batch_size", type=int, default=4, help='train_batch_size')
    parser.add_argument("-vbs", "--val_batch_size", type=int, default=4, help='val_batch_size')

    parser.add_argument("-spth", "--snapshot_path", type=str,
                        default='DenseNet121_change_avg', help='epoch')

    args = parser.parse_args()

    Image_size = args.Image_size
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    batch_size = val_batch_size
    workers = 4
    print(Image_size)
    print(train_batch_size)
    print(val_batch_size)

    model_snapshot_path = args.snapshot_path.replace('\n', '').replace('\r', '') + '/'
    kfold_path = '../data/fold_5_by_study_image/'

    df_test = pd.read_csv(test_csv_path)  
    c_test = list(set(df_test['filename'].values.tolist()))
    df_all = pd.read_csv(csv_path)
    is_center = False
    is_aug = True
    num_aug = 10
    val_truth_oof = df_all.sort_values(by='filename').reset_index(drop = True).loc[:, ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']].values

    backbone = args.backbone
    print(backbone)
    predict_all(backbone, Image_size)


