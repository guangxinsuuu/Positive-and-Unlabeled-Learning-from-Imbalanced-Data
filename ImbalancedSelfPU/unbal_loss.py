import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as func
from sklearn import metrics
from sklearn.metrics import auc
#from torch.autograd.grad_mode import F
# training loss, sigmoid, false negative loss, which should be non-negative


# training loss, sigmoid, oversampled PU
class OversampledPULossFunc(nn.Module):
    def __init__(self):
        super(OversampledPULossFunc, self).__init__()
        return

    def forward(self, output_p, output_n, prior, prior_prime):
        prior_prime=0.5
        cost = prior_prime * torch.mean(torch.sigmoid(-output_p))
        cost = cost + (1-prior_prime)/(1-prior)*torch.mean(torch.sigmoid(output_n))
        cost = cost -  (1-prior_prime)/(1-prior)*prior * torch.mean(torch.sigmoid(output_p))
        return cost

class OversampledNNPULossFunc(nn.Module):
    def __init__(self):
        super(OversampledNNPULossFunc, self).__init__()
        return

    def forward(self, output_p, output_n, prior, prior_prime):
        prior_prime=0.5
        cost = 0
        cost = cost + (1-prior_prime)/(1-prior)*torch.mean(torch.sigmoid(output_n))
        cost = cost -  (1-prior_prime)/(1-prior)*prior * torch.mean(torch.sigmoid(output_p))
        return cost

# training loss, sigmoid, PN
class PNTrainingSigmoid(nn.Module):
    def __init__(self):
        super(PNTrainingSigmoid, self).__init__()
        return

    def forward(self, output_p, output_n, prior):
        cost = prior * torch.mean(torch.sigmoid(-output_p))
        cost = cost + (1 - prior) * torch.mean(torch.sigmoid(output_n))
        return cost





# non-negative training loss, 0-1
class NNZeroOneTrain(nn.Module):
    def __init__(self):
        super(NNZeroOneTrain, self).__init__()
        return

    def forward(self, output_p, output_n, prior):
        cost = torch.mean((1 + torch.sign(output_n)) / 2) - prior * torch.mean((1 + torch.sign(output_p)) / 2)
        cost = max(cost, 0)
        cost = cost + prior * torch.mean((1 - torch.sign(output_p)) / 2)
        return cost


# test loss, 0-1
class ZeroOneTest(nn.Module):
    def __init__(self):
        super(ZeroOneTest, self).__init__()
        return

    def forward(self, output_p, output_n, prior):
        cost = prior * torch.mean((1 - torch.sign(output_p)) / 2)
        cost = cost + (1 - prior) * torch.mean((1 + torch.sign(output_n)) / 2)
        return cost

### precision, recall, F1 and AUC for test
class PUPrecision(nn.Module):
    def __init__(self):
        super(PUPrecision, self).__init__()
        return

    def forward(self, output_p, output_n, prior):
        true_positive = torch.sum((torch.sign(output_p) + 1)/2)
        all_predicted_positive = torch.sum((torch.sign(output_p) + 1)/2) + torch.sum((torch.sign(output_n)+1)/2)
        if all_predicted_positive == 0:
            precision = 0
        else:
            precision = float(true_positive)/float(all_predicted_positive)
        return precision

class PURecall(nn.Module):
    def __init__(self):
        super(PURecall, self).__init__()
        return

    def forward(self, output_p, output_n, prior):
        true_positive = torch.sum((torch.sign(output_p) + 1) / 2)
        all_real_positive = len(output_p)
        recall = float(true_positive) / float(all_real_positive)
        return recall


class PUF1(nn.Module):
    def __init__(self):
        super(PUF1, self).__init__()
        return

    def forward(self, output_p, output_n, prior):
        true_positive = torch.sum((torch.sign(output_p) + 1) / 2)
        all_predicted_positive = torch.sum((torch.sign(output_p) + 1) / 2)+ torch.sum((torch.sign(output_n) + 1) / 2)
        all_real_positive = len(output_p)
        if all_predicted_positive == 0:
            precision = 0
        else:
            precision = float(true_positive) / float(all_predicted_positive)
        recall = float(true_positive) / float(all_real_positive)
        if precision ==0 or recall == 0:
            F1 = 0
        else:
            F1 = 2*precision*recall/(precision+recall)
        return F1



class Auc_loss(nn.Module):
    def __init__(self):
        super(Auc_loss, self).__init__()
        return

    def forward(self, out_put, target):
        fpr, tpr, thresholds = metrics.roc_curve(target.cpu().numpy(),

                                                 out_put.cpu().numpy(), pos_label=1)

        Auc = metrics.auc(fpr, tpr)

        return Auc


