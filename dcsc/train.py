import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from preprocess import *
from evaluation import *
import scanpy as sp
import st_loss
from sklearn.cluster import KMeans

import time
import math
import sys
import argparse


import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
import torchvision

# import models

class ContrastiveRepresentation(nn.Module):
    """
    Clustering network
    Args:
        nn ([type]): [description]
    """
    def __init__(self, dims, dropout = 0.8):
        super(ContrastiveRepresentation, self).__init__()
        self.dims = dims
        self.n_stacks = len(self.dims) #- 1
        enc = []
        for i in range(self.n_stacks - 1):
            if i == 0:
                enc.append(nn.Dropout(p =dropout))
            enc.append(nn.Linear(self.dims[i], self.dims[i+1]))
            enc.append(nn.BatchNorm1d(self.dims[i+1]))
            enc.append(nn.ReLU())

        enc = enc[:-2]
        self.encoder= nn.Sequential(*enc)
        self._reset_prams()

    def _reset_prams(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        return

    def forward(self, x):
        latent_out = self.encoder(x)
        latent_out = F.normalize(latent_out, dim = 1)
        return latent_out
    
class STClustering(nn.Module):
    """
    Clustering network
    Args:
        nn ([type]): [description]
    """
    def __init__(self, dims, t_alpha = 1):
        super(STClustering, self).__init__()
        self.t_alpha = t_alpha
        self.phase = "1"
        self.dims = dims
        self.n_stacks = len(self.dims) - 1
        enc = []
        for i in range(self.n_stacks - 1):
            enc.append(nn.Linear(self.dims[i], self.dims[i+1]))
            enc.append(nn.BatchNorm1d(self.dims[i+1]))
            enc.append(nn.ReLU())

        enc = enc[:-2]
        self.encoder= nn.Sequential(*enc)
        self.clustering = ClusterlingLayer(self.dims[-2], self.dims[-1])
        self._reset_prams()

    def _reset_prams(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        return
    def cal_latent(self, hidden, alpha):
        sum_y = torch.sum(torch.mul(hidden, hidden), dim=1)
        num = -2.0 * torch.matmul(hidden, torch.t(hidden)) + sum_y.view((-1, 1)) + sum_y
        num = num / alpha
        num = torch.pow(1.0 + num, -(alpha + 1.0) / 2.0)
        zerodiag_num = num - torch.diag(torch.diagonal(num))
        latent_p = torch.t(torch.t(zerodiag_num) / torch.sum(zerodiag_num, dim=1))
        return num, latent_p
    
    def target_dis(self, latent_p):
        latent_q = torch.t(torch.t(torch.pow(latent_p, 2))/torch.sum(latent_p, dim = 1))
        res = torch.t(torch.t(latent_q)/torch.sum(latent_q, dim =1))
        return res

    def forward(self, x):
        latent_out = self.encoder(x)
        if self.phase == "1":
            latent_out = F.normalize(latent_out, dim = 1)
        if self.phase == "2":
            normalized = F.normalize(latent_out, dim = 1)
            latent_dist1, latent_dist2 = self.clustering(latent_out)
        
            num, latent_p = self.cal_latent(latent_out, 1)
            latent_q = self.target_dis(latent_p)
            latent_p = latent_p + torch.diag(torch.diagonal(num))
            latent_q = latent_q + torch.diag(torch.diagonal(num))
            result = {
                "latent": latent_out,
                "latent_dist1": latent_dist1,
                "latent_dist2": latent_dist2,
                "latent_q": latent_q,
                "latent_p": latent_p,
                "num": num,
                "normalized": normalized

            }
            return result
        return latent_out
    


class MeanAct(nn.Module):
    def __init__(self, minval=1e-5, maxval=1e6):
        '''
        Init method.
        '''
        super().__init__() # init the base class
        self.minval = minval
        self.maxval = maxval

    def forward(self, inp):
        return torch.clamp(torch.exp(inp), self.minval, self.maxval)
    
class DispAct(nn.Module):
    def __init__(self, minval=1e-4, maxval=1e4):
        '''
        Init method.
        '''
        super().__init__() # init the base class
        self.minval = minval
        self.maxval = maxval

    def forward(self, inp):
        return torch.clamp(F.softplus(inp), self.minval, self.maxval)
    
    
class ClusterlingLayer(nn.Module):
    """
    Clustering layer to be applied on top of the representation layer.
    Args:
        nn ([type]): [description]
    """
    def __init__(self, in_features=10, out_features=10, alpha=1.0):
        super(ClusterlingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.unsqueeze(1) - self.weight
        x = torch.mul(x, x)
        dist1 = torch.sum(x, dim=2)
        temp_dist1 = dist1 - torch.min(dist1, dim = 1)[0].view((-1, 1))
        q = torch.exp(-temp_dist1)
        q = torch.t(torch.t(q)/torch.sum(q, dim =1))
        q = torch.pow(q, 2)
        q = torch.t(torch.t(q)/torch.sum(q, dim =1))
        dist2 = dist1 * q
        return dist1, dist2


    def extra_repr(self):
        return 'in_features={}, out_features={}, alpha={}'.format(
            self.in_features, self.out_features, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)
    
def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) +np.inf, x)


def _nelem(x):
    nelem = torch.sum((~torch.isnan(x)).type(torch.FloatTensor))
    res = torch.where(torch.equal(nelem, 0.), 1., nelem)
    return res

def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return torch.divide(torch.sum(x), nelem)


def clustering_loss(result, gamma = 0.001, alpha = 0.001):
    """
    Clustering loss used in our method on top of the representation layer.
    Args:
        result ([type]): [description]
        gamma (float, optional): [description]. Defaults to 0.001.
        alpha (float, optional): [description]. Defaults to 0.001.
    Returns:
        [type]: [description]
    """
    cross_entropy_loss = - torch.sum(result['latent_q'] * torch.log(result['latent_p']))
    kmeans_loss = torch.mean(torch.sum(result['latent_dist2'], dim=1))
    entropy_loss = -torch.sum(result['latent_q'] * torch.log(result['latent_q']))
    kl_loss = cross_entropy_loss- entropy_loss
    total_loss =  alpha * kmeans_loss + gamma * kl_loss
    return total_loss


def adjust_learning_rate( optimizer, epoch, lr):
    p = {
      'epochs': 500,
     'optimizer': 'sgd',
     'optimizer_kwargs': {'nesterov': False,
              'weight_decay': 0.0001,
              'momentum': 0.9,

                         },
     'scheduler': 'cosine',
     'scheduler_kwargs': {'lr_decay_rate': 0.1},
     }

    
    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        new_lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2
         
    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            new_lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        new_lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return lr


def train_model(X,
                cluster_number,
                Y=None,
                nb_epochs=30,
                lr=0.4,
                temperature=0.07,
                dropout=0.9,
                evaluate_training = False,
                layers = [200, 40, 60],
                save_pred = False,
                noise = None,
                device = 'cpu'):
    """[summary]
    Args:
        X ([type]): [description]
        cluster_number ([type]): [description]
        Y ([type], optional): [description]. Defaults to None.
        nb_epochs (int, optional): [description]. Defaults to 20.
        lr ([type], optional): [description]. Defaults to 1e-5.
        temperature (float, optional): [description]. Defaults to 0.07.
        dropout (float, optional): [description]. Defaults to 0.8.
        evaluate_training (bool, optional): [description]. Defaults to False.
        layers (list, optional): [description]. Defaults to [256, 64, 32].
        save_pred (bool, optional): [description]. Defaults to False.
        noise ([type], optional): [description]. Defaults to None.
        use_cpu ([type], optional): [description]. Defaults to None.
    Returns:
        [type]: [description]
    """

    dims = np.concatenate([[X.shape[1]], layers])#[X.shape[1], 256, 64, 32]
    model = ContrastiveRepresentation(dims, dropout=dropout)
    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()),
                                 lr=lr)

    criterion_rep = st_loss.SupConLoss(temperature=temperature)
    batch_size = 200

    losses = []
    idx = np.arange(len(X))
    for epoch in range(nb_epochs):

        model.train()
        adjust_learning_rate(optimizer, epoch, lr)
        np.random.shuffle(idx)
        loss_ = 0
        for pre_index in range(len(X) // batch_size + 1):
            c_idx = np.arange(pre_index * batch_size,
                              min(len(X), (pre_index + 1) * batch_size))
            if len(c_idx) == 0:
                continue
            c_idx = idx[c_idx]
            c_inp = X[c_idx]
            if noise is None or noise ==0:
                # pdb.set_trace()
                #input1 = torch.FloatTensor(c_inp).to(device)
                #input2 = torch.FloatTensor(c_inp).to(device)
                input1 = c_inp.to(device)
                input2 = c_inp.to(device)
            else:
                #noise_vec = np.random.normal(loc = 0, scale = noise, size = c_inp.shape)
                #input1 = torch.FloatTensor(c_inp + noise_vec).to(device)
                #noise_vec = np.random.normal(loc = 0, scale = noise, size = c_inp.shape)
                #input2 = torch.FloatTensor(c_inp + noise_vec).to(device)
                
                noise_vec = np.random.normal(loc = 0, scale = noise, size = c_inp.shape)
                noise_vec = torch.tensor(noise_vec).to(torch.float32).to(device)
                input1 = (c_inp + noise_vec).to(device)
                noise_vec = np.random.normal(loc = 0, scale = noise, size = c_inp.shape)
                noise_vec = torch.tensor(noise_vec).to(torch.float32).to(device)
                input2 = (c_inp + noise_vec).to(device)


            anchors_output = model(input1)
            neighbors_output = model(input2)

            features = torch.cat(
                [anchors_output.unsqueeze(1),
                 neighbors_output.unsqueeze(1)],
                dim=1)
            total_loss = criterion_rep(features)
            loss_ += total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        if evaluate_training and Y is not None:
            model.eval()
            with torch.no_grad():
                result = model(X)
                features = result.detach().cpu().numpy()
            # pdb.set_trace()
            res = cluster_embedding(features, cluster_number, Y, save_pred = save_pred)
            print(f"{epoch}). Loss {loss_}, ARI {res['kmeans_ari']}, NMI {res['kmeans_nmi']}, HOMO {res['kmeans_homo']}, COMP {res['kmeans_comp']}")

        losses.append(loss_)
    model.eval()
    with torch.no_grad():
        result = model(X.to(device))
        features = result.detach().cpu().numpy()
    return features


def run_leiden(data, leiden_n_neighbors=300):
    """
    Performs Leiden community detection on given data.
    Args:
        data ([type]): [description]
        n_neighbors (int, optional): [description]. Defaults to 10.
        n_pcs (int, optional): [description]. Defaults to 40.
    Returns:
        [type]: [description]
    """
    import scanpy as sc
    n_pcs=0
    adata = sc.AnnData(data)
    sc.pp.neighbors(adata, n_neighbors=leiden_n_neighbors, n_pcs=n_pcs, use_rep='X')
    sc.tl.leiden(adata)
    pred = adata.obs['leiden'].to_list()
    pred = [int(x) for x in pred]
    return pred


def run(X,
        cluster_number,
        dataset,
        Y=None,
        nb_epochs=30,
        lr=0.4,
        temperature=0.07,
        dropout=0.9,
        layers = [200, 40, 60],
        save_to="data/",
        save_pred = False,
        noise = None,
        device = None,
        cluster_methods = ["KMeans", "Leiden"],
        evaluate_training = False,
        leiden_n_neighbors=300):
    """[summary]
    Args:
        X ([type]): [description]
        cluster_number ([type]): [description]
        dataset ([type]): [description]
        Y ([type], optional): [description]. Defaults to None.
        nb_epochs (int, optional): [description]. Defaults to 30.
        lr (float, optional): [description]. Defaults to 0.4.
        temperature (float, optional): [description]. Defaults to 0.07.
        dropout (float, optional): [description]. Defaults to 0.9.
        layers (list, optional): [description]. Defaults to [256, 64, 32].
        save_to (str, optional): [description]. Defaults to "data/".
        save_pred (bool, optional): [description]. Defaults to False.
        noise ([type], optional): [description]. Defaults to None.
        use_cpu ([type], optional): [description]. Defaults to None.
        evaluate_training (bool, optional): [description]. Defaults to False.
        leiden_n_neighbors (int, optional): [description]. Defaults to 300.
    """
    results = {}
    
    X = torch.tensor(X).to(torch.float32).to(device)

    start = time.time()
    embedding = train_model(X,
              cluster_number,
              Y=Y,
              nb_epochs=nb_epochs,
              lr=lr,
              temperature=temperature,
              dropout=dropout,
              layers = layers,
              evaluate_training=evaluate_training,
              save_pred= save_pred,
              noise = noise, 
              device = device)
    if save_pred:
        results[f"features"] = embedding
    elapsed = time.time() -start
    # spdb.set_trace()
    res_eval = cluster_embedding(embedding, cluster_number, Y, save_pred = save_pred,
                                 leiden_n_neighbors=leiden_n_neighbors, cluster_methods = cluster_methods)
    results = {**results, **res_eval}
    results["dataset"] = dataset
    results["time"] = elapsed
#     if os.path.isdir(save_to) == False:
#         os.makedirs(save_to)
#     with open(f"{save_to}/{dataset}.pickle", 'wb') as handle:
#         pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return results



def cluster_embedding(embedding, cluster_number, Y, save_pred = False, 
                      leiden_n_neighbors=300, cluster_methods =["KMeans", "Leiden"]):
    """[summary]
    Args:
        embedding ([type]): [description]
        cluster_number ([type]): [description]
        Y ([type]): [description]
        save_pred (bool, optional): [description]. Defaults to False.
        leiden_n_neighbors (int, optional): [description]. Defaults to 300.
    Returns:
        [type]: [description]
    """
    result = {"t_clust" : time.time()}
    if "KMeans" in cluster_methods:
        # evaluate K-Means
        kmeans = KMeans(n_clusters=cluster_number,
                        init="k-means++",
                        random_state=0)
        pred = kmeans.fit_predict(embedding)
        if Y is not None:
            result[f"kmeans_ari"] = adjusted_rand_score(Y, pred)
            result[f"kmeans_nmi"] = normalized_mutual_info_score(Y, pred)
            result[f"kmeans_comp"] = completeness_score(Y, pred)
            result[f"kmeans_homo"] = homogeneity_score(Y, pred)
        # result[f"kmeans_sil"] = silhouette_score(embedding, pred)
        # result[f"kmeans_cal"] = calinski_harabasz_score(embedding, pred)
        result["t_k"] = time.time()
        if save_pred:
            result[f"kmeans_pred"] = pred
    """
    if "Leiden" in cluster_methods:
        # evaluate leiden
        pred = run_leiden(embedding, leiden_n_neighbors)
        if Y is not None:
            result[f"leiden_ari"] = adjusted_rand_score(Y, pred)
            result[f"leiden_nmi"] = normalized_mutual_info_score(Y, pred)
        result[f"leiden_sil"] = silhouette_score(embedding, pred)
        result[f"leiden_cal"] = calinski_harabasz_score(embedding, pred)
        result["t_l"] = time.time()
        if save_pred:
            result[f"leiden_pred"] = pred
    """
    return result

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush() #每次写入后刷新到文件中，防止程序意外结束
    def flush(self):
        self.log.flush()

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", default = 'klein', type = str)
    parser.add_argument("--nb_epochs", default = 30, type = int)
    parser.add_argument("--lr", default = 0.4, type = float)
    parser.add_argument("--temperature", default = 0.07, type = float)
    parser.add_argument("--dropout", default = 0.9, type = float)
    parser.add_argument("--layers", default = '[512, 256, 64]', type = str)
    parser.add_argument("--noise", default = 1.0, type = float)
    parser.add_argument("--device", default = 'cuda:0', type = str)
    parser.add_argument("--times", default = '1', type = int)
    args = parser.parse_args()
    
    
    data_path = 'data/%s_2000.csv'% args.dataset
    label_path = 'data/%s_truelabel.csv' % args.dataset
    log_path = 'log/%s_%d.txt'%(args.dataset, args.times)
    
    # pretrain_path = 'model/%s_pretrain_param.pth' % args.dataset
    # model_path = 'model/%s_param.pth' % args.dataset
    
    ### load dataset
    sys.stdout = Logger(log_path)
    print(args)
    
    x = pd.read_csv(data_path, header=None).to_numpy().astype(np.float32)
    # y = pd.read_csv(label_path, header=None).squeeze()
    y = pd.read_csv(label_path)['x']
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
    
    run(
        X = adata.X,
        cluster_number=n_clusters,
        dataset='klein',
        Y=adata.obs['Group'],
        nb_epochs=args.nb_epochs,
        lr=args.lr,
        temperature=args.temperature,
        dropout=args.dropout,
        layers=eval(args.layers),
        noise=args.noise,
        device=args.device,
        evaluate_training=True)
