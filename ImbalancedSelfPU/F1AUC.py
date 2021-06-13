import sklearn
from torch import nn
from sklearn import metrics
from sklearn.metrics import roc_auc_score
class F1_loss(nn.Module):
    def __init__(self):
        super(F1_loss, self).__init__()
        return

    def forward(self, out_put, target):


        F1 = sklearn.metrics.f1_score(target.cpu().numpy(),
                                      out_put.cpu().numpy(),
                                      labels=None,
                                      pos_label=1,
                                      average='binary', sample_weight=None)
        return F1

class PUAUC(nn.Module):



    def __init__(self):
        super(PUAUC, self).__init__()
        return

    def forward(self, out_put, target):
        fpr, tpr, thresholds = metrics.roc_curve(target.cpu().numpy(),

                                                 out_put.cpu().numpy(), pos_label=1)

        Auc = metrics.auc(fpr, tpr)
        #Auc = roc_auc_score(target.cpu().numpy(),out_put.cpu().numpy())

        return Auc