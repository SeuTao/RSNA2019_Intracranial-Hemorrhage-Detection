import numpy as np
from sklearn.metrics.ranking import roc_auc_score
import torch
from sklearn.metrics import fbeta_score
import time
import cv2
from sklearn.metrics import log_loss
import torch


def computeAUROC(dataGT, dataPRED, classCount):

    outAUROC = []

    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))

    return outAUROC

def search_f1(output, target):
    max_result_f1_list = []
    max_threshold_list = []
    precision_list = []
    recall_list = []
    eps=1e-20
    target = target.type(torch.cuda.ByteTensor)

    # print(output.shape, target.shape)
    for i in range(output.shape[1]):

        output_class = output[:, i]
        target_class = target[:, i]
        max_result_f1 = 0
        max_threshold = 0

        optimal_precision = 0
        optimal_recall = 0

        for threshold in [x * 0.01 for x in range(0, 100)]:

            prob = output_class > threshold
            label = target_class > 0.5
            # print(prob, label)
            TP = (prob & label).sum().float()
            TN = ((~prob) & (~label)).sum().float()
            FP = (prob & (~label)).sum().float()
            FN = ((~prob) & label).sum().float()

            precision = TP / (TP + FP + eps)
            recall = TP / (TP + FN + eps)
            # print(precision, recall)
            result_f1 = 2 * precision  * recall / (precision + recall + eps)

            if result_f1.item() > max_result_f1:
                # print(max_result_f1, max_threshold)
                max_result_f1 = result_f1.item()
                max_threshold = threshold

                optimal_precision = precision
                optimal_recall = recall

        max_result_f1_list.append(round(max_result_f1,3))
        max_threshold_list.append(max_threshold)
        precision_list.append(round(optimal_precision.item(),3))
        recall_list.append(round(optimal_recall.item(),3))

    return max_threshold_list, max_result_f1_list, precision_list, recall_list

def weighted_log_loss(output, target, weight=[1,1,1,1,1,1]):

    loss = torch.nn.BCELoss()
    loss_list = []
    for i in range(output.shape[1]):
        output_class = output[:, i]
        target_class = target[:, i]
        loss_class = loss(output_class, target_class)   
        loss_list.append(float(loss_class.cpu().numpy()))

    loss_sum = np.mean(np.array(weight)*np.array(loss_list))
    loss_list = [round(x, 4) for x in loss_list]
    loss_sum = round(loss_sum, 4)

    return loss_list, loss_sum

def weighted_log_loss_numpy(output, target, weight=[1,1,1,1,1,1]):

    loss_list = []
    for i in range(output.shape[1]):
        output_class = output[:, i]
        target_class = target[:, i]
        loss_class = log_loss(target_class.ravel(), output_class.ravel(), eps=1e-7)
        loss_list.append(loss_class)

    loss_sum = np.mean(np.array(weight)*np.array(loss_list))
    loss_list = [round(x, 4) for x in loss_list]
    loss_sum = round(loss_sum, 4)

    return loss_list, loss_sum