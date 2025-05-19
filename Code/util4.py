from torch import nn
import math
import random
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_curve, f1_score
import os
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import matplotlib


def reset_parameters(named_parameters):
    for i in named_parameters():
        if len(i[1].size()) == 1:
            std = 1.0 / math.sqrt(i[1].size(0))
            nn.init.uniform_(i[1], -std, std)
        else:
            if 'weight_hh' in i[0]:
                nn.init.orthogonal_(i[1])
            else:
                nn.init.kaiming_normal_(i[1])

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def KS(y_true,y_proba):
    return ks_2samp(y_proba[y_true == 1], y_proba[y_true == 0]).statistic

def GM(y_true,y_pred):
    gmean = 1.0
    labels = sorted(list(set(y_true)))
    for label in labels:
        recall = (y_pred[y_true==label]==label).mean()
        gmean = gmean*recall
    return gmean**(1/len(labels))

def metrics(trues, preds):
    trues = np.array(trues).squeeze(2)
    trues = np.concatenate(trues,-1)
    preds = np.concatenate(preds,0)
    predict = preds.argmax(-1)
    acc = sum(preds.argmax(-1) == trues) / len(trues)
    recall_macro = recall_score(trues, predict, average='macro')
    ks = KS(trues,predict)
    gm = GM(trues,predict)

    eva_list = [acc,recall_macro,ks,gm]
    return eva_list

def train_metrics(trues, preds):
    trues = np.array(trues).squeeze(2)
    trues = np.concatenate(trues,-1)
    preds = np.concatenate(preds,0)
    predict = preds.argmax(-1)
    precision = precision_score(trues, predict, average='binary', pos_label=1)
    acc = sum(preds.argmax(-1) == trues) / len(trues)
    recall = recall_score(trues,predict, average='binary', pos_label=1)
    auc = roc_auc_score(trues,preds[:,1])
    return acc, recall,precision,auc




def dot(x, y, sparse=False):
    res = torch.matmul(x, y)
    return res

def ele_mul(x, y, sparse=False):
    if sparse:
        res = x.__mul__(y)
    else:
        res = torch.mul(x, y)
    return res


def Save_list(list1,filename):
    file2 = open(filename, 'w')
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            file2.write(str(list1[i][j]))
            file2.write(',')
        file2.write('\n')
    file2.close()
