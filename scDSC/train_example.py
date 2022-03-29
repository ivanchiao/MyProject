import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from utils import load_data, load_graph, RAdam
from GNN import GNNLayer
from evaluation import eva
from torch.utils.data import DataLoader, TensorDataset
import h5py
import scanpy as sc
from preprocess import read_dataset, normalize
from layers import ZINBLoss, MeanAct, DispAct
from torch.autograd import Variable
import os
from sklearn.metrics import adjusted_rand_score as ari_score
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from time import time as get_time

import pandas as pd
import pdb

import warnings

warnings.filterwarnings('ignore')


class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z1, n_z2, n_z3):
        super(AE, self).__init__()

        self.enc_1 = Linear(n_input, n_enc_1)
        self.BN1 = nn.BatchNorm1d(n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.BN2 = nn.BatchNorm1d(n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.BN3 = nn.BatchNorm1d(n_enc_3)

        self.z1_layer = Linear(n_enc_3, n_z1)
        self.BN4 = nn.BatchNorm1d(n_z1)
        self.z2_layer = Linear(n_z1, n_z2)
        self.BN5 = nn.BatchNorm1d(n_z2)
        self.z3_layer = Linear(n_z2, n_z3)
        self.BN6 = nn.BatchNorm1d(n_z3)

        self.dec_1 = Linear(n_z3, n_dec_1)
        self.BN7 = nn.BatchNorm1d(n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.BN8 = nn.BatchNorm1d(n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.BN9 = nn.BatchNorm1d(n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)


    def forward(self, x):
        enc_h1 = F.relu(self.BN1(self.enc_1(x)))
        enc_h2 = F.relu(self.BN2(self.enc_2(enc_h1)))
        enc_h3 = F.relu(self.BN3(self.enc_3(enc_h2)))

        z1 = self.BN4(self.z1_layer(enc_h3))
        z2 = self.BN5(self.z2_layer(z1))
        z3 = self.BN6(self.z3_layer(z2))

        dec_h1 = F.relu(self.BN7(self.dec_1(z3)))
        dec_h2 = F.relu(self.BN8(self.dec_2(dec_h1)))
        dec_h3 = F.relu(self.BN9(self.dec_3(dec_h2)))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z3, z2, z1, dec_h3


class SDCN(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z1, n_z2, n_z3, n_clusters, v=1):
        super(SDCN, self).__init__()
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,

            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,

            n_input=n_input,
            n_z1=n_z1,
            n_z2=n_z2,
            n_z3=n_z3,
        )
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z1)
        self.gnn_5 = GNNLayer(n_z1, n_z2)
        self.gnn_6 = GNNLayer(n_z2, n_z3)
        self.gnn_7 = GNNLayer(n_z3, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z3))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self._dec_mean = nn.Sequential(nn.Linear(n_dec_3, n_input), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(n_dec_3, n_input), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(n_dec_3, n_input), nn.Sigmoid())
        # degree
        self.v = v
        self.zinb_loss = ZINBLoss().cuda()

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z3, z2, z1, dec_h3 = self.ae(x)

        sigma = 0.5
        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
        h = self.gnn_5((1 - sigma) * h + sigma * z1, adj)
        h = self.gnn_6((1 - sigma) * h + sigma * z2, adj)
        h = self.gnn_7((1 - sigma) * h + sigma * z3, adj, active=False)

        predict = F.softmax(h, dim=1)

        _mean = self._dec_mean(dec_h3)
        _disp = self._dec_disp(dec_h3)
        _pi = self._dec_pi(dec_h3)
        zinb_loss = self.zinb_loss

        q = 1.0 / (1.0 + torch.sum(torch.pow(z3.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z3, _mean, _disp, _pi, zinb_loss



def target_distribution(q):

    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn(dataset, X_raw, sf):
    global p
    model = SDCN(
                 n_enc_1=args.n_enc_1,
                 n_enc_2=args.n_enc_2,
                 n_enc_3=args.n_enc_3,
                 n_dec_1=args.n_dec_1,
                 n_dec_2=args.n_dec_2,
                 n_dec_3=args.n_dec_3,
                 n_input=args.n_input,
                 n_z1=args.n_z1,
                 n_z2=args.n_z2,
                 n_z3=args.n_z3,
                 n_clusters=args.n_clusters,
                 v=1).to(device)
    print(model)
    # optimizer = Adam(model.parameters(), lr=args.lr)
    optimizer = RAdam(model.parameters(), lr=args.lr)

    adj = load_graph(args.graph, dataset.x.shape[0], args.k)
    print(args.k)
    adj = adj.cuda()


    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

    with torch.no_grad():
        _, _, _, _, z, _, _, _ = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=args.n_init)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 0)

    for epoch in range(Para[2]):
        if epoch % 1 == 0:
            _, tmp_q, pred, _, _, _, _, _ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            res2 = pred.data.cpu().numpy().argmax(1)  # Z
            res3 = p.data.cpu().numpy().argmax(1)  # P
            # eva(y, res1, str(epoch) + 'Q')
            # eva(y, res2, str(epoch) + 'Z')
            # eva(y, res3, str(epoch) + 'P')
            eva(y, res2, epoch)

        
        x_bar, q, pred, z, meanbatch, dispbatch, pibatch, zinb_loss = model(data, adj)

        binary_crossentropy_loss = F.binary_cross_entropy(q, p)
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)
        X_raw = torch.tensor(X_raw).cuda()
        sf = torch.tensor(sf).cuda()
        # pdb.set_trace()
        zinb_loss = zinb_loss(X_raw, meanbatch, dispbatch, pibatch, sf)
        loss = Balance_para[0] * binary_crossentropy_loss + Balance_para[1] * ce_loss + Balance_para[2] * re_loss +Balance_para[3] * zinb_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
################################# 此处需要修改 ######################################
File = ['data/klein_2000.csv', "data/klein_truelabel.csv", "graph/kelin_graph.txt", 'model/klein_pretrain.pkl']
################################# 此处需要修改 ######################################

# x_df = pd.read_csv(File[0]).T
# x_df = x_df.iloc[1:, :]
# x = x_df.to_numpy().astype(np.float32)
x = pd.read_csv(File[0], header=None).to_numpy().astype(np.float32)
y_df = pd.read_csv(File[1])
y = y_df['x']

lab = y.unique().tolist()
ind = list(range(0, len(lab)))
mapping = {j: i for i, j in zip(ind, lab)}
y = y.map(mapping).to_numpy()

time_start = get_time()

model_para = [1000, 1000, 4000]
Para = [2048, 1e-3, 1000]
Cluster_para = [np.unique(y).shape[0], 2000, 500, 10, x.shape[1], 20]
Balance_para = [0.1, 0.01, 1, 0.1, 0.5]

class Args(object):
    def __init__(self):
        
        self.name = File[0]
        self.graph = File[2]
        self.pretrain_path = File[3]
        
        self.n_enc_1 = model_para[0]
        self.n_enc_2 = model_para[1]
        self.n_enc_3 = model_para[2]
        
        self.n_dec_1 = model_para[2]
        self.n_dec_2 = model_para[1]
        self.n_dec_3 = model_para[0]
        
        self.k = None
        self.lr = Para[1]
        
        self.n_clusters = Cluster_para[0]
        self.n_z1 = Cluster_para[1]
        self.n_z2 = Cluster_para[2]
        self.n_z3 = Cluster_para[3]
        self.n_input = Cluster_para[4]
        self.n_init = Cluster_para[5]

args = Args()
        
args.cuda = torch.cuda.is_available()
print("use cuda: {}".format(args.cuda))
device = torch.device("cuda" if args.cuda else "cpu")

args.pretrain_path = File[3]
dataset = load_data(x, y)

adata = sc.AnnData(x)
adata.obs['Group'] = y
adata = read_dataset(adata,
                     transpose=False,
                     test_split=False,
                     copy=True)

adata = normalize(adata,
                  filter_min_counts=False,
                  size_factors=True,
                  normalize_input=True,
                  logtrans_input=True)

X = adata.X
X_raw = adata.raw.X
sf = adata.obs.size_factors
dataset1 = TensorDataset(torch.Tensor(X), torch.Tensor(X_raw), torch.Tensor(sf))
dataloader = DataLoader(dataset1, batch_size=Para[0], shuffle=True)
train_sdcn(dataset, X_raw, sf)
time = get_time() - time_start
print("Running Time:" + str(time))
