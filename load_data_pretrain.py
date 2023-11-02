import numpy as np
from collections import defaultdict
import json

class Data:
    def __init__(self, data_dir):
        self.reg2id = self.load_reg(data_dir)
        self.ent2id, self.rel2id, self.kg_data = self.load_kg(data_dir)
        self.nreg = len(self.reg2id)

        print('number of node=%d, number of edge=%d, number of relations=%d' % (len(self.ent2id), len(self.kg_data), len(self.rel2id)))
        print('region num={}'.format(len(self.reg2id)))
        print('load finished..')

    def load_reg(self, data_dir):      
        with open(data_dir + 'region2info.json', 'r') as f:
            region2info = json.load(f)

        regions = sorted(region2info.keys(), key = lambda x:x)
        reg2id = dict([(x,i) for i,x in enumerate(regions)])

        return reg2id

    def load_kg(self, data_dir):
        ent2id, rel2id = self.reg2id.copy(), {}
        kg_data_str = []
        with open(data_dir + 'kg.txt', 'r') as f:
            for line in f.readlines(): 
                h,r,t = line.strip().split('\t')
                kg_data_str.append((h,r,t))
        ents = sorted(list(set([x[0] for x in kg_data_str] + [x[2] for x in kg_data_str])))
        rels = sorted(list(set([x[1] for x in kg_data_str])))
        for i, x in enumerate(ents):
            try:
                ent2id[x]
            except KeyError:
                ent2id[x] = len(ent2id)
        rel2id = dict([(x, i) for i, x in enumerate(rels)])
        kg_data = [[ent2id[x[0]], rel2id[x[1]], ent2id[x[2]]] for x in kg_data_str]
        
        return ent2id, rel2id, kg_data
    
        