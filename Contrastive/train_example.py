import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scanpy as sp
import pandas as pd
import numpy as np

from network import Network
from contrastive_loss import *
from torch.utils.data import TensorDataset, DataLoader
from evaluation import evaluate
import pdb


def buildNetwork(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        # net.append(nn.Dropout(0.2))
    return nn.Sequential(*net)


class AE(nn.Module):

    def __init__(self, input_dim, z_dim, encodeLayer=[], activation="relu", sigma=2.5, device='cpu'):

        super(AE, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.encoder = buildNetwork([input_dim]+encodeLayer, type="encode", activation=activation)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self.to(device)

    def forward(self, x):
        out = self.encoder(x + torch.randn_like(x) * sigma)
        out = self._enc_mu(out)
        return out


def drop(x, p=0.2):
    mask_list = [torch.rand(x.shape[1]) < p for _ in range(x.shape[0])]
    mask = torch.vstack(mask_list)
    new_x = x.clone()
    new_x[mask] = 0.0
    return new_x


def train_model(dataset,
                model,
                batch_size=256,
                lr=0.001,
                i_temperature=0.5,
                c_temperature=1.0,
                num_epochs=600,
                num_workers=0,
                p=0.2,
                device='cpu'):
    
    trainlaoder = DataLoader(dataset, shuffle=True,  batch_size=batch_size, num_workers=num_workers)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_instance_loss = 0.0
        total_cluster_loss = 0.0
        model.train()
        for batch_idx, (x, _) in enumerate(trainlaoder):

            iloss= InstanceLoss(batch_size=x.shape[0], temperature=i_temperature, device=device)
            closs = ClusterLoss(class_num=n_cluster, temperature=c_temperature, device=device)

            x = x.to(torch.float32).to(device)
            x1 = drop(x, p)
            x2 = drop(x, p)
            z_i, z_j, c_i, c_j = model.forward(x1, x2)

            instance_loss = iloss(z_i, z_j)
            cluster_loss = closs(c_i, c_j)

            loss = instance_loss + cluster_loss
            # loss = instance_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_instance_loss += instance_loss.item()
            total_cluster_loss += cluster_loss.item()

        print("Epoch {%d}, instance_loss {%.4f}, cluster_loss {%.4f}" % (epoch, total_instance_loss / len(trainlaoder),
                                                                         total_cluster_loss / len(trainlaoder)))

        model.eval()
        with torch.no_grad():
            y_pred = model.forward_cluster(dataset.tensors[0].to(torch.float32).to(device))
            y_pred = y_pred.detach().cpu().numpy()
            y_truth = dataset.tensors[1].numpy()

            evaluate(y_truth, y_pred, epoch=epoch)


#%%
file_path = '../sc/data/klein_2000.csv'
label_path = '../sc/data/klein_truelabel.csv'
sigma = 2.5
num_epochs = 600
num_workers=6
lr=0.001
device = 'cuda:0'

batch_size = 64
encodeLayer = [512, 256, 128]
feature_dim = 32
z_dim = 64
p = 0.2
i_temperature=0.5
c_temperature=1.0


#%%
x = pd.read_csv(file_path, header=None).to_numpy().astype(np.float32)
# y = pd.read_csv(label_path, header=None).squeeze()
y = pd.read_csv(label_path)['x']

#%%
lab = y.unique().tolist()
ind = list(range(0, len(lab)))
mapping = {j: i for i, j in zip(ind, lab)}
y = y.map(mapping).to_numpy()

n_cluster = len(ind)
dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y))

#%%
encoder = AE(
    input_dim=x.shape[1],
    z_dim=z_dim,
    encodeLayer=encodeLayer,
    sigma=sigma,
    device=device)


contrastive_net = Network(
    net=encoder,
    feature_dim=feature_dim,
    class_num=n_cluster,
    device=device)

train_model(
    dataset=dataset,
    model=contrastive_net,
    batch_size=batch_size,
    lr=lr,
    i_temperature=i_temperature,
    c_temperature=c_temperature,
    num_epochs=num_epochs,
    p=p,
    num_workers=num_workers,
    device=device
)

