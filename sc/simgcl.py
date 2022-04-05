# -*- coding: utf-8 -*-
"""

@author: LMC_ZC
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class SimGCL(nn.Module):
    
    def __init__(self, n_users, n_items, norm_adj, emb_size, n_layers, l2_reg, ssl_reg, tau, eps, device):
        
        super(SimGCL, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        
        self.norm_adj = norm_adj.to(device)
        self.emb_size = emb_size
        self.l2_reg = l2_reg
        self.ssl_reg = ssl_reg
        self.tau = tau
        self.eps = eps
        
        self.n_layers = n_layers
        self.device = device
        
        self._create_embeddings()
        self.to(self.device)
        
    def _create_embeddings(self):

        self.embeddings = nn.ModuleDict()
        self.embeddings['user_embeddings'] = nn.Embedding(self.n_users, self.emb_size)
        self.embeddings['item_embeddings'] = nn.Embedding(self.n_items, self.emb_size)
        
        nn.init.normal_(self.embeddings['user_embeddings'].weight, std=0.001)
        nn.init.normal_(self.embeddings['item_embeddings'].weight, std=0.001)
        
    def propagate(self, adj, user_emb, item_emb):
        ego_embeddings = torch.cat([user_emb, item_emb], dim=0)
        all_embeddings = [ego_embeddings]
        
        for k in range(1, self.n_layers+1):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings
    
    def simgcl_propagate(self, adj, user_emb, item_emb):
        ego_embeddings = torch.cat([user_emb, item_emb], dim=0)
        all_embeddings = [ego_embeddings]
        
        for k in range(1, self.n_layers+1):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            random_noise = torch.rand(ego_embeddings.shape).to(self.device)
            ego_embeddings += torch.multiply(torch.sign(ego_embeddings), F.normalize(random_noise, p=2, dim=1)) * self.eps
            all_embeddings += [ego_embeddings]
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings
    
    def calc_bpr_loss(self, u_g_embeddings, i_g_embeddings, users, pos_items, neg_items):

        batch_u_embeddings = u_g_embeddings[users]
        batch_pos_i_embeddings = i_g_embeddings[pos_items]
        batch_neg_i_embeddings = i_g_embeddings[neg_items]

        regularizer = (1 / 2) * (batch_u_embeddings.norm(2).pow(2) + batch_pos_i_embeddings.norm(2).pow(2) +
                                 batch_neg_i_embeddings.norm(2).pow(2)) / batch_u_embeddings.shape[0]
        emb_loss = self.l2_reg * regularizer
        pos_score = torch.sum(batch_u_embeddings * batch_pos_i_embeddings, dim=1)
        neg_score = torch.sum(batch_u_embeddings * batch_neg_i_embeddings, dim=1)
        bpr_loss = -1.0 * F.logsigmoid(pos_score - neg_score).sum()

        return bpr_loss, emb_loss
    
    def calc_ssl_loss(self, user_emb1, item_emb1, user_emb2, item_emb2, users, pos_items):
        
        p_user_emb1 = user_emb1[torch.unique(users)]
        p_item_emb1 = item_emb1[torch.unique(pos_items)]
        p_user_emb2 = user_emb2[torch.unique(users)]
        p_item_emb2 = item_emb2[torch.unique(pos_items)]
        
        normalize_emb_user1 = F.normalize(p_user_emb1, p=2, dim=1)
        normalize_emb_user2 = F.normalize(p_user_emb2, p=2, dim=1)
        normalize_emb_item1 = F.normalize(p_item_emb1, p=2, dim=1)
        normalize_emb_item2 = F.normalize(p_item_emb2, p=2, dim=1)

        pos_score_u = torch.sum(torch.multiply(normalize_emb_user1, normalize_emb_user2), dim=1)
        pos_score_i = torch.sum(torch.multiply(normalize_emb_item1, normalize_emb_item2), dim=1)
        ttl_score_u = torch.matmul(normalize_emb_user1, normalize_emb_user2.T)
        ttl_score_i = torch.matmul(normalize_emb_item1, normalize_emb_item2.T)
        
        pos_score_u = torch.exp(pos_score_u / self.tau)
        ttl_score_u = torch.sum(torch.exp(ttl_score_u / self.tau), axis=1)
        pos_score_i = torch.exp(pos_score_i / self.tau)
        ttl_score_i = torch.sum(torch.exp(ttl_score_i / self.tau), axis=1)
        ssl_loss = -torch.log(pos_score_u / ttl_score_u).sum() - torch.log(pos_score_i / ttl_score_i).sum()
        
        return self.ssl_reg * ssl_loss