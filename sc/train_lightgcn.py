# -*- coding: utf-8 -*-
"""
@author: LMC_ZC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from simgcl import SimGCL
from lightgcn import LightGCN

import pandas as pd
import numpy as np

from dataset import Dataset
from graph import Graph
from evaluation import eva
from loader import BPRTrainLoader

##### parameters
data_path = 'data/baron1_2000.csv'
label_path = 'data/baron1_truelabel.csv'
n_layers = 3
device='cuda:0'
num_epochs=100
eps=0.1
tau=0.2
ssl_reg=0.1
l2_reg=0.001
emb_size=256
batch_size=4096
num_workers=6
lr=0.001

print(data_path)
print(label_path)
print(n_layers)

print(l2_reg)
print(emb_size)
print(batch_size)
print(lr)

dataset = Dataset(data_path=data_path, label_path=label_path)
graph = Graph(n_users=dataset.n_cell, n_items=dataset.n_gene, train_u2i=dataset.train_u2i)
norm_adj = graph.generate_ori_norm_adj()


# baseline: kmeans
"""
for k in range(1, 10+1):
    kmeans = KMeans(n_clusters=np.unique(dataset.label).size, n_init=10).fit(dataset.data)
    label = dataset.label.astype(np.float32)
    preds = kmeans.labels_.astype(np.float32)
    eva(label, preds, epoch=k)
"""

############################## lightgcn ##############################
model = LightGCN(
    n_users=dataset.n_cell,
    n_items=dataset.n_gene,
    norm_adj=norm_adj,
    emb_size=emb_size,
    n_layers=n_layers,
    l2_reg=l2_reg,
    device='cuda:0')

optimizer = optim.Adam(params=model.parameters(), lr=lr)
train_tensor = BPRTrainLoader(
    train_set=dataset.train_set,
    train_u2i=dataset.train_u2i,
    n_items=dataset.n_gene)
loader = DataLoader(train_tensor, shuffle=True, num_workers=num_workers, batch_size=batch_size)
for epoch in range(num_epochs):
    result = {
            'loss': 0.0,
            'bpr_loss': 0.0,
            'reg_loss': 0.0}
    
    model.train()
    for uij in loader:
        u = uij[0].type(torch.long).to(device)
        i = uij[1].type(torch.long).to(device)
        j = uij[2].type(torch.long).to(device)
        
        main_user_emb, main_item_emb = model.propagate(
            model.norm_adj,
            model.embeddings['user_embeddings'].weight,
            model.embeddings['item_embeddings'].weight)

        bpr_loss, reg_loss = model.calc_bpr_loss(main_user_emb, main_item_emb, u, i, j)
        loss = bpr_loss + reg_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        result['loss'] += loss.item()
        result['bpr_loss'] += bpr_loss.item()
        result['reg_loss'] += reg_loss.item()
    
    result['loss'] = result['loss'] / len(loader)
    result['bpr_loss'] = result['bpr_loss'] / len(loader)
    result['reg_loss'] = result['reg_loss'] / len(loader)
    
    eval_res = 'epoch [%d] ' % epoch    
    for name, value in result.items():
        eval_res += name + ':' + '[%.6f]' % value + ' '
    print(eval_res)
    
    model.eval()
    user_emb, item_emb = model.propagate(model.norm_adj,
                                         model.embeddings['user_embeddings'].weight,
                                         model.embeddings['item_embeddings'].weight)
    
    user_emb = user_emb.detach().cpu().numpy()
    item_emb = item_emb.detach().cpu().numpy()
    
    ##### eval1
    rating = np.matmul(user_emb, item_emb.T)
    inputs_emb = np.concatenate([rating, dataset.data], axis=1)
    kmeans = KMeans(n_clusters=np.unique(dataset.label).size).fit(inputs_emb)
    label = dataset.label.astype(np.float32)
    preds = kmeans.labels_.astype(np.float32)
    eva(label, preds, epoch)
    
    ##### eval2
    inputs_emb = np.concatenate([user_emb, dataset.data], axis=1)
    kmeans = KMeans(n_clusters=np.unique(dataset.label).size).fit(inputs_emb)
    label = dataset.label.astype(np.float32)
    preds = kmeans.labels_.astype(np.float32)
    eva(label, preds, epoch)
