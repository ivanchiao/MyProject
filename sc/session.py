# -*- coding: utf-8 -*-
"""

@author: LMC_ZC
"""

import torch

class Session(object):

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def train(self, loader, optimizer):
        result = {
            'loss': 0.0,
            'bpr_loss': 0.0,
            'reg_loss': 0.0,
            'ssl_loss': 0.0
        }

        self.model.train()
        for uij in loader:
            u = uij[0].type(torch.long).to(self.device)
            i = uij[1].type(torch.long).to(self.device)
            j = uij[2].type(torch.long).to(self.device)
            
            
            main_user_emb, main_item_emb = self.model.propagate(
                self.model.norm_adj,
                self.model.embeddings['user_embeddings'].weight,
                self.model.embeddings['item_embeddings'].weight)
            
            ssl_user_emb1, ssl_item_emb1 = self.model.simgcl_propagate(
                self.model.norm_adj,
                self.model.embeddings['user_embeddings'].weight,
                self.model.embeddings['item_embeddings'].weight)
            
            ssl_user_emb2, ssl_item_emb2 = self.model.simgcl_propagate(
                self.model.norm_adj,
                self.model.embeddings['user_embeddings'].weight,
                self.model.embeddings['item_embeddings'].weight)
            
            bpr_loss, reg_loss = self.model.calc_bpr_loss(main_user_emb, main_item_emb, u, i, j)
            ssl_loss = self.model.calc_ssl_loss(ssl_user_emb1, ssl_item_emb1, ssl_user_emb2, ssl_item_emb2, u, i)
            loss = bpr_loss + reg_loss + ssl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            result['loss'] += loss.item()
            result['bpr_loss'] += bpr_loss.item()
            result['reg_loss'] += reg_loss.item()
            result['ssl_loss'] += ssl_loss.item()

        result['loss'] = result['loss'] / len(loader)
        result['bpr_loss'] = result['bpr_loss'] / len(loader)
        result['reg_loss'] = result['reg_loss'] / len(loader)
        result['ssl_loss'] = result['ssl_loss'] / len(loader)
        return result