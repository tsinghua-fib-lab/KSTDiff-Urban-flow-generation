import torch
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
import torch.nn as nn
import numpy as np



class MyLoss_Pretrain(torch.nn.Module):
    def __init__(self):
        super(MyLoss_Pretrain, self).__init__()
        return

    def forward(self, pred, tar):
        kg_pred_max = torch.max(pred, dim=1)[0].view(-1, 1)
        kg_pred_log_max_sum = torch.log(torch.sum(torch.exp(pred-kg_pred_max), dim=1)).view(-1, 1)
        kg_pred_log_softmax = pred - kg_pred_max - kg_pred_log_max_sum
        loss_kge = - kg_pred_log_softmax[tar==True].mean()
        return loss_kge



class TuckER(torch.nn.Module):
    def __init__(self, d, d1, **kwargs):
        super(TuckER, self).__init__()
        d2 = d1
        device = kwargs['device']
        self.E = torch.nn.Embedding(len(d.ent2id), d1)
        self.R = torch.nn.Embedding(len(d.rel2id), d2)
        self.init()
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)), dtype=torch.float, requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs['dropout_in'])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs['dropout_h1'])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs['dropout_h2'])
        self.loss = MyLoss_Pretrain()
        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)

        self.device = device

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, h_idx, r_idx):
        h = self.E(h_idx)
        x = self.bn0(h)
        x = self.input_dropout(x)
        x = x.view(-1, 1, h.size(1))

        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, h.size(1), h.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, h.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)

        pred = torch.mm(x, self.E.weight.transpose(1, 0))
        return pred