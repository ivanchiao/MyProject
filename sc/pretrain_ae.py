import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
from sklearn.cluster import KMeans
from evaluation import eva

import pandas as pd
import numpy as np


class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z1, n_z2, n_z3, device):
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

        self.to(device)

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

        return x_bar,  z3


def pretrain_ae(model, dataset, y):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            x = torch.Tensor(dataset.x).to(device).to(torch.float32)
            x_bar, z3 = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))

            kmeans = KMeans(n_clusters=np.unique(y).shape[0], n_init=20).fit(z3.data.cpu().numpy())
            eva(y, kmeans.labels_, epoch)
            # Generate a pre-trained model
    torch.save(model.state_dict(), model_path)


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


file_path = 'data/klein_2000.csv'
label_path = 'data/klein_truelabel.csv'
model_path = 'model/klein_param.pkl'
batch_size = 1024
num_workers = 6
lr = 1e-4
device = 'cuda:0'
num_epochs = 300

# load dataset
x = pd.read_csv(file_path, header=None).to_numpy().astype(np.float32)
y = pd.read_csv(label_path)['x']
lab = y.unique().tolist()
ind = list(range(0, len(lab)))
mapping = {j: i for i, j in zip(ind, lab)}
y = y.map(mapping).to_numpy()


model = AE(
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
    device='cuda:0')

dataset = LoadDataset(x)
pretrain_ae(model, dataset, y)