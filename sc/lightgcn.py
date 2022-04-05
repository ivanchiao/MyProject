# -*- coding: utf-8 -*-
"""

@author: LMC_ZC
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    
    def __init__(self, n_users, n_items, norm_adj, emb_size, l2_reg, n_layers, device):
        
        super(LightGCN, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        
        self.norm_adj = norm_adj.to(device)
        self.emb_size = emb_size
        self.l2_reg = l2_reg
        self.n_layers = n_layers
        self.device = device
        
        self._create_embeddings()
        self.to(self.device)
        
    def _create_embeddings(self):

        self.embeddings = nn.ModuleDict()
        self.embeddings['user_embeddings'] = nn.Embedding(self.n_users, self.emb_size)
        self.embeddings['item_embeddings'] = nn.Embedding(self.n_items, self.emb_size)

        nn.init.xavier_uniform_(self.embeddings['user_embeddings'].weight)
        nn.init.xavier_uniform_(self.embeddings['item_embeddings'].weight)
    
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

    def calc_bpr_loss(self, user_emb, item_emb, users, pos_items, neg_items):

        batch_u_embeddings = user_emb[users]
        batch_pos_i_embeddings = item_emb[pos_items]
        batch_neg_i_embeddings = item_emb[neg_items]

        regularizer = (1 / 2) * (batch_u_embeddings.norm(2).pow(2) + batch_pos_i_embeddings.norm(2).pow(2) +
                                 batch_neg_i_embeddings.norm(2).pow(2)) / batch_u_embeddings.shape[0]
        emb_loss = self.l2_reg * regularizer
        pos_score = torch.sum(batch_u_embeddings * batch_pos_i_embeddings, dim=1)
        neg_score = torch.sum(batch_u_embeddings * batch_neg_i_embeddings, dim=1)
        bpr_loss = -1.0 * F.logsigmoid(pos_score - neg_score).sum()

        return bpr_loss, emb_loss