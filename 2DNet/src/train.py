import os
import time
import pandas as pd
import gc
import cv2
import csv
import random
from sklearn.metrics.ranking import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR
import torch.utils.data

import torch.utils.data as data
from net.models import *
from dataset.dataset import *
from tuils.tools import *
from tuils.lrs_scheduler import WarmRestart, warm_restart, AdamW, RAdam
from tuils.loss_function import *
import torch.nn.functional as F
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(1992)
torch.cuda.manual_seed(1992)
np.random.seed(1992)
random.seed(1992)
from PIL import ImageFile
import sklearn
import copy
torch.backends.cudnn.benchmark = True
import argparse

def epochVal(model, dataLoader, loss_cls, c_val, val_batch_size):
    model.eval ()
    lossValNorm = 0
    valLoss = 0

    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    for i, (input, target) in enumerate (dataLoader):
        if i == 0:
            ss_time = time.time()
        print(str(i) + '/' + str(int(len(c_val)/val_batch_size)) + '     ' + str((time.time()-ss_time)/(i+1)), end='\r')
        target = target.view(-1, 6).contiguous().cuda()
        outGT = torch.cat((outGT, target), 0)
        varInput = torch.autograd.Variable(input)
        varTarget = torch.autograd.Variable(target.contiguous().cuda())
        varOutput = model(varInput)
        lossvalue = loss_cls(varOutput, varTarget)
        valLoss = valLoss + lossvalue.item()
        varOutput = varOutput.sigmoid()

        outPRED = torch.cat((outPRED, varOutput.data), 0)
        lossValNorm += 1

    valLoss = valLoss / lossValNorm

    auc = computeAUROC(outGT, outPRED, 6)
    auc = [round(x, 4) for x in auc]
    loss_list, loss_sum = weighted_log_loss(outPRED, outGT)

    return valLoss, auc, loss_list, loss_sum

def train(model_name, image_size):

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    header = ['Epoch', 'Learning rate', 'Time', 'Train Loss', 'Val Loss']

    if not os.path.isfile(snapshot_path + '/log.csv'):
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    df_all = pd.read_csv(csv_path)

    kfold_path_train = '../data/fold_5_by_study/'
    kfold_path_val = '../data/fold_5_by_study_image/'

    for num_fold in range(5):
        print('fold_num:',num_fold)

        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([num_fold]) 

        f_train = open(kfold_path_train + 'fold' + str(num_fold) + '/train.txt', 'r')
        f_val = open(kfold_path_val + 'fold' + str(num_fold) + '/val.txt', 'r')
        c_train = f_train.readlines()
        c_val = f_val.readlines()
        f_train.close()
        f_val.close()
        c_train = [s.replace('\n', '') for s in c_train]
        c_val = [s.replace('\n', '') for s in c_val]     

        # for debug
        # c_train = c_train[0:1000]
        # c_val = c_val[0:4000]

        print('train dataset study num:', len(c_train), '  val dataset image num:', len(c_val))
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['train dataset:', len(c_train), '  val dataset:', len(c_val)])  
            writer.writerow(['train_batch_size:', train_batch_size, 'val_batch_size:', val_batch_size])  

        train_transform, val_transform = generate_transforms(image_size)
        train_loader, val_loader = generate_dataset_loader(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers)

        model = eval(model_name+'()')
        model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002)
        scheduler = WarmRestart(optimizer, T_max=5, T_mult=1, eta_min=1e-5)
        model = torch.nn.DataParallel(model)
        loss_cls = torch.nn.BCEWithLogitsLoss(pos_weight = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).cuda())

        trMaxEpoch = 80
        for epochID in range (0, trMaxEpoch):
            epochID = epochID + 0

            start_time = time.time()
            model.train()
            trainLoss = 0
            lossTrainNorm = 10

            if epochID < 10:
                pass
            elif epochID < 80:
                if epochID != 10:
                    scheduler.step()
                    scheduler = warm_restart(scheduler, T_mult=2) 
            else:
                optimizer.param_groups[0]['lr'] = 1e-5

            for batchID, (input, target) in enumerate (train_loader):
                if batchID == 0:
                    ss_time = time.time()

                print(str(batchID) + '/' + str(int(len(c_train)/train_batch_size)) + '     ' + str((time.time()-ss_time)/(batchID+1)), end='\r')
                varInput = torch.autograd.Variable(input)
                target = target.view(-1, 6).contiguous().cuda()
                varTarget = torch.autograd.Variable(target.contiguous().cuda())
                varOutput = model(varInput)
                lossvalue = loss_cls(varOutput, varTarget)
                trainLoss = trainLoss + lossvalue.item()
                lossTrainNorm = lossTrainNorm + 1

                lossvalue.backward()
                optimizer.step()
                optimizer.zero_grad()
                del lossvalue

            trainLoss = trainLoss / lossTrainNorm

            if (epochID+1)%5 == 0 or epochID > 79 or epochID == 0:
                valLoss, auc, loss_list, loss_sum = epochVal(model, val_loader, loss_cls, c_val, val_batch_size)
        
            epoch_time = time.time() - start_time

            if (epochID+1)%5 == 0 or epochID > 79:
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'valLoss': valLoss}, snapshot_path + '/model_epoch_' + str(epochID) + '_' + str(num_fold) + '.pth')

            result = [epochID,
                      round(optimizer.state_dict()['param_groups'][0]['lr'], 6),
                      round(epoch_time, 0),
                      round(trainLoss, 5),
                      round(valLoss, 5),
                      'auc:', auc,
                      'loss:',loss_list,
                      loss_sum]

            print(result)

            with open(snapshot_path + '/log.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result)  

        del model

def valid_snapshot(model_name, image_size):
    dir = r'./DenseNet121_change_avg_256'
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    header = ['Epoch', 'Learning rate', 'Time', 'Train Loss', 'Val Loss']

    if not os.path.isfile(snapshot_path + '/log.csv'):
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    df_all = pd.read_csv(csv_path)

    kfold_path_val = '../data/fold_5_by_study_image/'
    loss_cls = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).cuda())
    for num_fold in range(5):
        print('fold_num:', num_fold)

        ckpt = r'model_epoch_best_'+str(num_fold)+'.pth'
        ckpt = os.path.join(dir,ckpt)

        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([num_fold])

        f_val = open(kfold_path_val + 'fold' + str(num_fold) + '/val.txt', 'r')
        c_val = f_val.readlines()
        f_val.close()
        c_val = [s.replace('\n', '') for s in c_val]

        print('  val dataset image num:', len(c_val))

        val_transform = albumentations.Compose([
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0,
                                     p=1.0)
        ])

        val_dataset = RSNA_Dataset_val_by_study_context(df_all, c_val, val_transform)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False)

        model = eval(model_name + '()')
        model = model.cuda()
        model = torch.nn.DataParallel(model)

        if ckpt is not None:
            print(ckpt)
            model.load_state_dict(torch.load(ckpt, map_location=lambda storage, loc: storage)["state_dict"])

        valLoss, auc, loss_list, loss_sum = epochVal(model, val_loader, loss_cls, c_val, val_batch_size)

        result = [round(valLoss, 5),
                  'auc:', auc,
                  'loss:', loss_list,
                  loss_sum]

        with open(ckpt + '_log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result)
        print(result)


if __name__ == '__main__':
    csv_path = '../data/stage1_train_cls.csv'
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-backbone", "--backbone", type=str, default='DenseNet121_change_avg', help='backbone')
    parser.add_argument("-img_size", "--Image_size", type=int, default=256, help='image_size')
    parser.add_argument("-tbs", "--train_batch_size", type=int, default=32, help='train_batch_size')
    parser.add_argument("-vbs", "--val_batch_size", type=int, default=32, help='val_batch_size')
    parser.add_argument("-save_path", "--model_save_path", type=str,
                        default='DenseNet169_change_avg', help='epoch')
    args = parser.parse_args()

    Image_size = args.Image_size
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    workers = 24
    backbone = args.backbone
    print(backbone)
    print('image size:', Image_size)
    print('train batch size:', train_batch_size)
    print('val batch size:', val_batch_size)
    snapshot_path = 'data_test/' + args.model_save_path.replace('\n', '').replace('\r', '')
    train(backbone, Image_size)
    # valid_snapshot(backbone, Image_size)
