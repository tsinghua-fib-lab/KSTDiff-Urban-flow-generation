import os
import json
import numpy as np
import datetime
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

dataset='nyc' ##################### modify the dataset here
filepath='./data/data_{}/'.format(dataset)  
resultpath = './output/output_{}/'.format(dataset)
assert os.path.exists(resultpath)


with open(filepath+"region2info.json",'r') as f:
    region2info=json.load(f)
regions=sorted(region2info.keys(),key=lambda x:x)
reg2id=dict([(x,i) for i,x in enumerate(regions)])
region2info=dict([(k,region2info[k]) for k in regions])


class MaximumMeanDiscrepancy_numpy(object):
    """calculate MMD"""

    def __init__(self):
        super(MaximumMeanDiscrepancy_numpy, self).__init__()

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

        n_samples = source.shape[0]+ target.shape[0]
        total = np.concatenate([source, target], axis=0)  # 合并在一起
        total0 = np.expand_dims(total, axis = 0)
        total0 = np.tile(total0, (total.shape[0], 1, 1))

        total1 = np.expand_dims(total, axis = 1)
        total1 = np.tile(total1, (1, total.shape[0], 1))

        L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|

        # 计算多核中每个核的bandwidth
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = np.sum(L2_distance) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]

        # 高斯核的公式，exp(-|x-y|/bandwith)
        kernel_val = [np.exp(-L2_distance / bandwidth_temp) for
                      bandwidth_temp in bandwidth_list]

        return sum(kernel_val)  # 将多个核合并在一起

    def __call__(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n = source.shape[0]
        m = target.shape[0]

        kernels = self.guassian_kernel(
            source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:n, :n]
        YY = kernels[n:, n:]
        XY = kernels[:n, n:]
        YX = kernels[n:, :n]

        # K_ss矩阵，Source<->Source
        XX = np.divide(XX, n * n).sum(axis=1).reshape(1, -1)
        # K_st矩阵，Source<->Target
        XY = np.divide(XY, -n * m).sum(axis=1).reshape(1, -1)

        # K_ts矩阵,Target<->Source
        YX = np.divide(YX, -m * n).sum(axis=1).reshape(1, -1)
        # K_tt矩阵,Target<->Target
        YY = np.divide(YY, m * m).sum(axis=1).reshape(1, -1)

        loss = (XX + XY).sum() + (YX + YY).sum()
        return loss


with open(filepath + "alldayflow.json",'r') as f:
    alldayflow=json.load(f)

def is_weekday(datestr): # 20160101
    date=datetime.datetime.strptime(datestr, "%Y%m%d")
    return date.weekday() in [0,1,2,3,4]

allday_data=[]
for k,v in alldayflow.items():
    if is_weekday(k):
        allday_data.append(v)
allday_data=np.array(allday_data)

M=np.max(allday_data)
m=np.min(allday_data)

allday_flow=np.mean(allday_data,0)

# load train and test regions
with open(filepath + 'train_regs.json', 'r') as f:
    trainregs = json.load(f)
with open(filepath + 'test_regs.json', 'r') as f:
    testregs = json.load(f)
    
trainids = [reg2id[x] for x in trainregs]
sampids = [reg2id[x] for x in testregs]

test_reg_flow = allday_flow[sampids,:,:]
test_flow = allday_data[:,sampids,:,:]


def cal_smape(p_pred, p_real, eps=0.00000001):
    out=np.mean(np.abs(p_real - p_pred) / ((np.abs(p_real) + np.abs(p_pred)) / 2 + eps))
    return out

it=500
pred=np.load(resultpath+"sample_{}_final.npz".format(it))

pred=pred['sample']
pred=(pred*(M-m)+m+M)/2 

pred_flow=np.mean(pred,0)
rmse=metrics.mean_squared_error(pred_flow.flatten(),test_reg_flow.flatten(),squared=False)
mae=metrics.mean_absolute_error(pred_flow.flatten(),test_reg_flow.flatten())
smape=cal_smape(pred_flow.flatten(),test_reg_flow.flatten())

mmd = MaximumMeanDiscrepancy_numpy()
tmpmmds=[]
for i in range(pred.shape[1]):
    realflow=test_flow[:,i,:,:]
    genflow=pred[:,i,:,:]
    data_1=realflow.reshape(realflow.shape[0],-1)
    data_2=genflow.reshape(genflow.shape[0],-1)
    tmpmmds.append(mmd(data_1, data_2))
mmd=np.mean(tmpmmds)
print('%.2f\t%.2f\t%.2f\t%.2f'%(mae,rmse,smape,mmd))

