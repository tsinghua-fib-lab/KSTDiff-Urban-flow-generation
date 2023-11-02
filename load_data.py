import numpy as np
from collections import defaultdict
import json
import datetime

def is_weekday(datestr): # 20160101
    date=datetime.datetime.strptime(datestr, "%Y%m%d")
    return date.weekday() in [0,1,2,3,4]

class Data:
    def __init__(self, data_dir):
        self.reg2id, self.trainids, self.sampids, self.trainregs, self.sampleregs, self.regfeas = self.load_reg(data_dir)
        self.nreg = len(self.reg2id)
        self.ent2id, self.rel2id, self.kg_data, self.train_ent2id, self.trainkg_data, self.sample_ent2id, self.samplekg_data = self.load_kg(data_dir)
        self.train_data, self.min_data, self.max_data = self.load_flow(data_dir) # ndays*nreg*nhour*2
        self.features, self.scale, self.scale_pred_data, self.scale_pred_X, self.KGE_pretrain = self.load_pretrain(data_dir)
        

        print('number of node=%d, number of edge=%d, number of relations=%d' % (len(self.ent2id), len(self.kg_data), len(self.rel2id)))
        print('region num={}'.format(len(self.reg2id)))
        print('train data={}'.format(len(self.train_data)))
        print('load finished..')

    def load_reg(self, data_dir):      
        with open(data_dir + 'region2info.json', 'r') as f:
            region2info = json.load(f)

        regions = sorted(region2info.keys(), key = lambda x:x)
        reg2id = dict([(x,i) for i,x in enumerate(regions)])

        with open(data_dir + 'train_regs.json', 'r') as f:
            trainregs = json.load(f)

        with open(data_dir + 'test_regs.json', 'r') as f:
            testregs = json.load(f)

        sampregs = testregs
        trainids = [reg2id[x] for x in trainregs]
        sampids = [reg2id[x] for x in testregs]

        regfeas = []
        for r in regions:
            tmp = region2info[r]['feature']
            regfeas.append(tmp)
        regfeas = np.array(regfeas, dtype = np.float)
        
        return reg2id, trainids, sampids, trainregs, sampregs, regfeas.tolist()
    
    def load_kg(self, data_dir):
        ent2id, rel2id = self.reg2id.copy(), {}
        kg_data_str = []
        trainkg_str = []
        samplekg_str = []
        with open(data_dir + 'kg.txt', 'r') as f:
            for line in f.readlines(): 
                h,r,t = line.strip().split('\t')
                kg_data_str.append((h,r,t))
                # train/sample kg
                if h in self.trainregs and t in self.trainregs:
                    trainkg_str.append((h,r,t))
                if h in self.sampleregs and t in self.sampleregs:
                    samplekg_str.append((h,r,t))
        ents = sorted(list(set([x[0] for x in kg_data_str] + [x[2] for x in kg_data_str])))
        rels = sorted(list(set([x[1] for x in kg_data_str])))
        for i, x in enumerate(ents):
            try:
                ent2id[x]
            except KeyError:
                ent2id[x] = len(ent2id)
        rel2id = dict([(x, i) for i, x in enumerate(rels)])
        kg_data = [[ent2id[x[0]], rel2id[x[1]], ent2id[x[2]]] for x in kg_data_str]
        
        # train kg
        train_ent2id = dict([(x,i) for i,x in enumerate(self.trainregs)])
        trainkg_data = [[train_ent2id[x[0]], rel2id[x[1]], train_ent2id[x[2]]] for x in trainkg_str]
        # sample kg
        sample_ent2id = dict([(x,i) for i,x in enumerate(self.sampleregs)])
        samplekg_data = [[sample_ent2id[x[0]], rel2id[x[1]], sample_ent2id[x[2]]] for x in samplekg_str]

        return ent2id, rel2id, kg_data, train_ent2id, trainkg_data, sample_ent2id, samplekg_data

    def load_flow(self, data_dir):
        with open(data_dir+'alldayflow.json','r') as f:
            date2flowmat = json.load(f)
        train_data = []
        for k,v in date2flowmat.items():
            if is_weekday(k):
                train_data.append(v)
        train_data = np.array(train_data)
        M, m = np.max(train_data), np.min(train_data)
        train_data = (2 * train_data - m - M) / (M - m)


        return train_data.tolist(), m, M

    def load_pretrain(self,data_dir):
        data=np.load(data_dir+'ER.npz')
        KGE_pretrain = data['E_pretrain'] # nreg*kgedim

        scale = np.array(self.train_data) # nday*nreg*nhour*2
        scale = np.mean(scale, axis = 0) # nreg*nhour*2
        scale = scale.reshape(self.nreg, -1) # nreg*(nhour*2)
        scale = np.mean(scale, axis = 1, keepdims = 1) # nreg*1
        
        scale_pred_data = []
        scale_pred_X = []
        for i in range(self.nreg):
            X = self.regfeas[i]
            scale_pred_data.append([X, scale[i].item()])
            scale_pred_X.append(X)
        features = np.array(scale_pred_X)

        return features, scale, scale_pred_data, scale_pred_X, KGE_pretrain
        