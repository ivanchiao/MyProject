import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
from sklearn.cluster import KMeans
from evaluation import eva
from preprocess import *
from layers import *
from utils import RAdam

import pandas as pd
import numpy as np
import scanpy as sc


class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z1, n_z2, n_z3):
        super(AE, self).__init__()

        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.BN1 = nn.BatchNorm1d(n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.BN2 = nn.BatchNorm1d(n_enc_2)
        self.enc_3 = nn.Linear(n_enc_2, n_enc_3)
        self.BN3 = nn.BatchNorm1d(n_enc_3)

        self.z1_layer = nn.Linear(n_enc_3, n_z1)
        self.BN4 = nn.BatchNorm1d(n_z1)
        self.z2_layer = nn.Linear(n_z1, n_z2)
        self.BN5 = nn.BatchNorm1d(n_z2)
        self.z3_layer = nn.Linear(n_z2, n_z3)
        self.BN6 = nn.BatchNorm1d(n_z3)

        self.dec_1 = nn.Linear(n_z3, n_dec_1)
        self.BN7 = nn.BatchNorm1d(n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
        self.BN8 = nn.BatchNorm1d(n_dec_2)
        self.dec_3 = nn.Linear(n_dec_2, n_dec_3)
        self.BN9 = nn.BatchNorm1d(n_dec_3)
        self.x_bar_layer = nn.Linear(n_dec_3, n_input)

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
            n_z3=n_z3)
        # self.ae.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.ae.load_state_dict(torch.load(model_path))

        # cluster layer
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z3))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self._dec_mean = nn.Sequential(nn.Linear(n_dec_3, n_input), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(n_dec_3, n_input), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(n_dec_3, n_input), nn.Sigmoid())
        # degree
        self.v = v
        self.zinb_loss = ZINBLoss()

        self.to(device)

    def forward(self, x):
        # DNN Module
        x_bar, tra1, tra2, tra3, z3, z2, z1, dec_h3 = self.ae(x)

        _mean = self._dec_mean(dec_h3)
        _disp = self._dec_disp(dec_h3)
        _pi = self._dec_pi(dec_h3)
        zinb_loss = self.zinb_loss

        q = 1.0 / (1.0 + torch.sum(torch.pow(z3.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, z3, _mean, _disp, _pi, zinb_loss


def target_distribution(q):

    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn(x, y, X_raw, sf):
    global p
    model = SDCN(
        n_enc_1=1000,
        n_enc_2=1000,
        n_enc_3=4000,
        n_dec_1=4000,
        n_dec_2=1000,
        n_dec_3=1000,
        n_input=x.shape[1],
        n_z1=2000,
        n_z2=500,
        n_z3=10,
        n_clusters=n_clusters,
        v=1).to(device)
    print(model)
    # optimizer = Adam(model.parameters(), lr=args.lr)
    optimizer = RAdam(model.parameters(), lr=lr)
    data = torch.Tensor(x).to(device)

    X_raw = torch.tensor(X_raw).cuda()
    sf = torch.tensor(sf).cuda()

    with torch.no_grad():
        _, _, _, _, z, _, _, _ = model.ae(data)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 0)

    for epoch in range(num_epochs):
        if epoch % 1 == 0:
            _, tmp_q,  _, _, _, _, _ = model(data)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
            res = tmp_q.cpu().numpy().argmax(1)
            eva(y, res, epoch)
        x_bar, q, z, meanbatch, dispbatch, pibatch, zinb_loss = model(data)

        binary_crossentropy_loss = F.binary_cross_entropy(q, p)
        re_loss = F.mse_loss(x_bar, data)

        # pdb.set_trace()
        zinb_loss = zinb_loss(X_raw, meanbatch, dispbatch, pibatch, sf)
        loss = 0.1 * binary_crossentropy_loss + 1.0 * re_loss + 0.5 * zinb_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

file_path = 'data/chen_2000.csv'
label_path = 'data/chen_truelabel.csv'
model_path = 'model/chen_param.pkl'
batch_size = 2048
num_workers = 6
lr = 1e-4
device = 'cuda:0'
num_epochs = 1000

# load dataset
x = pd.read_csv(file_path, header=None).to_numpy().astype(np.float32)
y = pd.read_csv(label_path)['x']
lab = y.unique().tolist()
ind = list(range(0, len(lab)))
mapping = {j: i for i, j in zip(ind, lab)}
y = y.map(mapping).to_numpy()

n_clusters = np.unique(y).shape[0]

adata = sc.AnnData(x)
adata.obs['Group'] = y
adata = read_dataset(adata,
                     transpose=False,
                     test_split=False,
                     copy=True)

adata = normalize(adata,
                  filter_min_counts=False,
                  size_factors=True,
                  normalize_input=False,
                  logtrans_input=False)

X = adata.X
X_raw = adata.raw.X
sf = adata.obs.size_factors

# dataset1 = TensorDataset(torch.Tensor(X), torch.Tensor(X_raw), torch.Tensor(sf))
# loader1 = DataLoader(dataset1, shuffle=True, batch_size=batch_size, num_workers=num_workers)
train_sdcn(x, y, X_raw, sf)