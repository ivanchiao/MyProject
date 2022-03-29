import pdb
from time import time

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

import scanpy as sc

from preprocess import read_dataset, normalize
from scDCC import scDCC

import sklearn.metrics as metrics
from utils import cluster_acc


class Args(object):
    
    def __init__(self, ):
        
        self.data_path = './data/biase_2000.csv'
        self.label_path = './data/biase_truelabel.csv'
        self.n_clusters = 4
        self.gamma = 1.0
        self.batch_size = 64
        self.pretrain_epochs = 300
        self.ae_weight_file = 'AE_weights_p0_1.pth.tar'
        self.update_interval = 1
        self.save_dir = 'save/'
        self.tol = 0.001
        self.maxiter = 2000
        self.ae_weights = None

args = Args()

### load dataset

x = pd.read_csv(args.data_path, header=None).to_numpy().astype(np.float32)

# x_df = pd.read_csv(args.data_path).T
# x_df = x_df.iloc[1:, :]
# x = x_df.to_numpy().astype(np.float32)

y_df = pd.read_csv(args.label_path)
y = y_df['x']

lab = y.unique().tolist()
ind = list(range(0, len(lab)))
mapping = {j: i for i, j in zip(ind, lab)}
y = y.map(mapping).to_numpy()


# pdb.set_trace()
### preprocess dataset
adata = sc.AnnData(x)
adata.obs['Group'] = y

adata = read_dataset(adata)
adata = normalize(adata)

input_size = adata.n_vars
print(args)
print(adata.X.shape)
print(y.shape)

print(args.__dict__)

x_sd = adata.X.std(0)
x_sd_median = np.median(x_sd)
print("median of gene sd: %.5f" % x_sd_median)

ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array([]), np.array([]), np.array([]), np.array([])

sd = 2.5

model = scDCC(input_dim=adata.n_vars, z_dim=32, n_clusters=args.n_clusters, 
                encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=sd, gamma=args.gamma).cuda()
print(model)
t0 = time()

if args.ae_weights is None:
    model.pretrain_autoencoder(x=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, 
                                batch_size=args.batch_size, epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
else:
    if os.path.isfile(args.ae_weights):
        print("==> loading checkpoint '{}'".format(args.ae_weights))
        checkpoint = torch.load(args.ae_weights)
        model.load_state_dict(checkpoint['ae_state_dict'])
    else:
        print("==> no checkpoint found at '{}'".format(args.ae_weights))
        raise ValueError
print('Pretraining time: %d seconds.' % int(time() - t0))

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    
y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, sf=adata.obs.size_factors, y=y, batch_size=args.batch_size, num_epochs=args.maxiter, 
            ml_ind1=ml_ind1, ml_ind2=ml_ind2, cl_ind1=cl_ind1, cl_ind2=cl_ind2,
            update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)
acc = np.round(cluster_acc(y, y_pred), 5)
nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
print('Evaluating cells: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))