from load_data_pretrain import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model_pretrain import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import setproctitle
import mlflow
from mlflow.tracking import MlflowClient
import os
from tqdm import tqdm
import json
import copy
import random

import os
# os.environ['CUDA_VISIBLE_DEVICES']='7'
import setproctitle
setproctitle.setproctitle('TuckER_pretrain@zzl')

device = torch.device('cuda')

class Experiment:
    def __init__(self, lr, edim, batch_size, dr):
        self.lr = lr
        self.edim = edim
        self.batch_size = batch_size
        self.dr = dr
        self.num_iterations = args.num_iterations
        self.kwargs = params
        self.kwargs['device'] = device

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = torch.zeros((len(batch), len(d.ent2id)), device=device)
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        return torch.tensor(batch, dtype=torch.long, device=device), targets

    def train_and_eval(self):
        print('building model....')
        model = TuckER(d, self.edim, **self.kwargs)
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        if self.dr:
            scheduler = ExponentialLR(opt, self.dr)

        er_vocab = self.get_er_vocab(d.kg_data)
        er_vocab_pairs = list(er_vocab.keys())

        E_epoch, R_epoch, loss_epoch = [], [], []
        print("Starting training...")
        for it in range(1, self.num_iterations + 1):
            print('\n=============== Epoch %d Starts...===============' % it)
            start_train = time.time()
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in tqdm(range(0, len(er_vocab_pairs), self.batch_size)):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                h_idx = data_batch[:, 0]
                r_idx = data_batch[:, 1]
                predictions = model.forward(h_idx, r_idx)
                opt.zero_grad()
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.dr:
                scheduler.step()
            print('\nEpoch=%d, train time cost %.4fs, loss:%.8f' % (it, time.time() - start_train, np.mean(losses)))
            loss_epoch.append(np.mean(losses))
            mlflow.log_metrics({'train_time': time.time()-start_train, 'loss': loss_epoch[-1], 'current_it': it}, step=it)

        E_save=model.E.weight
        np.savez(archive_path + 'ER.npz',
                E_pretrain=E_save.detach().cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=200, nargs="?", help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=1024, nargs="?", help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.003, nargs="?", help="Learning rate.")
    parser.add_argument("--dr", type=float, default=0.995, nargs="?", help="Decay rate.")
    parser.add_argument("--edim", type=int, default=64, nargs="?", help="Entity embedding dimensionality.")
    parser.add_argument("--dropout_h1", type=float, default=0.2, nargs="?", help="Dropout rate.")
    parser.add_argument("--dropout_h2", type=float, default=0.3, nargs="?", help="Dropout rate.")
    parser.add_argument("--dropout_in", type=float, default=0.3, nargs="?", help="Dropout rate.")

    parser.add_argument("--exp_name", type=str, default="pretrain")
    parser.add_argument("--patience", type=int, default=50, nargs="?", help="valid patience.")
    parser.add_argument("--seed", type=int, default=20, nargs="?", help="random seed.")
    parser.add_argument("--model_name", type=str, default="TuckER")
    parser.add_argument("--loss", type=str, default="CE")

    parser.add_argument("--dataset", type=str, default='nyc', nargs="?", help="")

    args = parser.parse_args()
    print(args)

    data_dir = "./data/data_{}/".format(args.dataset)
    archive_path = './data/data_{}/'.format(args.dataset)
    
    assert os.path.exists(data_dir)
    if not os.path.exists(archive_path):
        os.makedirs(archive_path)

    # ~~~~~~~~~~~~~~~~~~ mlflow experiment ~~~~~~~~~~~~~~~~~~~~~

    experiment_name = 'TuckER_pretrain'
    mlflow.set_tracking_uri('/data1/zhouzhilun/flow_generation/mlflow_output/')
    client = MlflowClient()
    try:
        EXP_ID = client.create_experiment(experiment_name)
        print('Initial Create!')
    except:
        experiments = client.get_experiment_by_name(experiment_name)
        EXP_ID = experiments.experiment_id
        print('Experiment Exists, Continuing')
    with mlflow.start_run(experiment_id=EXP_ID) as current_run:
        
        # ~~~~~~~~~~~~~~~~~ reproduce setting ~~~~~~~~~~~~~~~~~~~~~
        seed = args.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        print('Loading data....')
        d = Data(data_dir=data_dir)
        params = vars(args)
        mlflow.log_params(params)
 
        experiment = Experiment(batch_size=args.batch_size, lr=args.lr, dr=args.dr, edim=args.edim)
        experiment.train_and_eval()

