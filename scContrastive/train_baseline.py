import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from layers import ZINBLoss, MeanAct, DispAct
import numpy as np
from sklearn.cluster import KMeans
import math, os
from sklearn import metrics
import pandas as pd
import scanpy as sp
from evaluation import evaluate
from preprocess import *
from collections import defaultdict
from sklearn import preprocessing
import random

def buildNetwork(layers, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)


class MyModel(nn.Module):
    def __init__(self, input_dim, z_dim, n_clusters, encodeLayer=[], decodeLayer=[], 
            activation="relu", sigma=1.0, alpha=1.0, gamma=1.0, device='cpu'):
        super(MyModel, self).__init__()
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.encoder = buildNetwork([input_dim]+encodeLayer, activation=activation)
        self.decoder = buildNetwork([z_dim]+decodeLayer, activation=activation)
        self.decoder1 = buildNetwork([z_dim]+decodeLayer, activation=activation)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())

        self.mu = nn.Parameter(torch.Tensor(n_clusters, z_dim))
        self.zinb_loss = ZINBLoss()
        self.to(device)
    
    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q
    
    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()
    
    def forward(self, x):
        h = self.encoder(x+torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        
        h1 = self.decoder1(z)
        h = (h1 + h) / 2
        
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        q = self.soft_assign(z0)
        return z0, q, _mean, _disp, _pi
    
    def encodeBatch(self, X, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        
        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs = Variable(xbatch)
            z,_, _, _, _ = self.forward(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
        kldloss = kld(p, q)
        return self.gamma*kldloss

    def pretrain_autoencoder(self, x, X_raw, size_factor, batch_size=256, lr=0.001, epochs=400, ae_save=True, ae_weights='AE_weights.pth.tar'):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        for epoch in range(epochs):
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch).cuda()
                x_raw_tensor = Variable(x_raw_batch).cuda()
                sf_tensor = Variable(sf_batch).cuda()
                _, _, mean_tensor, disp_tensor, pi_tensor = self.forward(x_tensor)
                loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=sf_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Pretrain epoch [{}/{}], ZINB loss:{:.4f}'.format(batch_idx+1, epoch+1, loss.item()))
        
        if ae_save:
            torch.save(self.state_dict(), ae_weights)

    def fit(self, X, X_raw, sf, y=None, lr=1., batch_size=256, num_epochs=10, save_path=''):
        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("Clustering stage")
        X = torch.tensor(X).cuda()
        X_raw = torch.tensor(X_raw).cuda()
        sf = torch.tensor(sf).cuda()
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)

        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(self.n_clusters, n_init=20)
        data = self.encodeBatch(X)
        self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
        self.y_pred_last = self.y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        if y is not None:
            acc, f1, nmi, ari, homo, comp = evaluate(y, self.y_pred)
            print('Initializing k-means: ACC= %.4f, F1= %.4f, NMI= %.4f, ARI= %.4f, HOMO= %.4f, COMP= %.4f' % (acc, f1, nmi, ari, homo, comp))
        
        self.train()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        best_ari = 0.0
        for epoch in range(num_epochs):
            # update the targe distribution p
            latent = self.encodeBatch(X)
            q = self.soft_assign(latent)
            p = self.target_distribution(q).data

            # evalute the clustering performance
            self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            acc, f1, nmi, ari, homo, comp = evaluate(y, self.y_pred)
            
            print('Cluster %d : ACC= %.4f, F1= %.4f, NMI= %.4f, ARI= %.4f, HOMO= %.4f, COMP= %.4f' % (epoch+1, acc, f1, nmi, ari, homo, comp))
            
            if best_ari < ari:
                best_ari = ari
                torch.save({'latent': latent, 'q': q, 'p': p}, save_path)
                print('save_successful')

            # train 1 epoch for clustering loss
            train_loss = 0.0
            recon_loss_val = 0.0
            cluster_loss_val = 0.0
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                xrawbatch = X_raw[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sfbatch = sf[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                pbatch = p[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                optimizer.zero_grad()
                inputs = Variable(xbatch)
                rawinputs = Variable(xrawbatch)
                sfinputs = Variable(sfbatch)
                target = Variable(pbatch)

                z, qbatch, meanbatch, dispbatch, pibatch = self.forward(inputs)

                cluster_loss = self.cluster_loss(target, qbatch)
                recon_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs)
                loss = cluster_loss + recon_loss
                loss.backward()
                optimizer.step()
                cluster_loss_val += cluster_loss.data * len(inputs)
                recon_loss_val += recon_loss.data * len(inputs)
                train_loss = cluster_loss_val + recon_loss_val

            print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f ZINB Loss: %.4f" % (
                epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num))

data_name = 'slyper'
data_path = 'data/%s_2000.csv'% data_name
label_path = 'data/%s_truelabel.csv' % data_name
pretrain_path = 'model/%s_pretrain_param.pth' % data_name
model_path = 'model/%s_param.pth' % data_name
x = pd.read_csv(data_path, header=None).to_numpy().astype(np.float32)
y = pd.read_csv(label_path, header=None).squeeze()
# y = pd.read_csv(label_path)['x']
lab = y.unique().tolist()
ind = list(range(0, len(lab)))
mapping = {j: i for i, j in zip(ind, lab)}
y = y.map(mapping).to_numpy()

adata = sc.AnnData(x)
adata.obs['Group'] = y
adata = read_dataset(adata, transpose=False, test_split=False, copy=False)
adata = normalize(adata, filter_min_counts=False, size_factors=True, normalize_input=False, logtrans_input=False)

input_size = adata.n_vars
n_clusters = adata.obs['Group'].unique().shape[0]

model = MyModel(
    input_dim=input_size,
    z_dim=32,
    n_clusters=n_clusters,
    encodeLayer=[512, 256, 64],
    decodeLayer=[64, 256, 512],
    activation='relu',
    sigma=2.5,
    alpha=1.0,
    gamma=1.0,
    device='cuda:0')

model.pretrain_autoencoder(
    x=adata.X,
    X_raw=adata.raw.X,
    size_factor=adata.obs.size_factors,
    batch_size=256,
    epochs=20,
    ae_weights=pretrain_path)

model.fit(
    X=adata.X,
    X_raw=adata.raw.X,
    sf=adata.obs.size_factors,
    y=y,
    lr=1.0,
    batch_size=256,
    num_epochs=100,
    save_path=model_path)