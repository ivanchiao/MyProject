# -*- coding: utf-8 -*-
"""

@author: LMC_ZC
"""


import torch
import pandas as pd
import numpy as np
from collections import defaultdict

class Dataset(object):
    
    def __init__(self, data_path, label_path):
        
        self.data_path = data_path
        self.label_path = label_path
        
        self.data = pd.read_csv(self.data_path, header=None).to_numpy()
        self.label = pd.read_csv(self.label_path)['x']
        
        self.n_cell = self.data.shape[0]
        self.n_gene = self.data.shape[1]
        
        self.train_set = self.build_edge()
        self.train_u2i = self.build_train_u2i(self.train_set)
        self.build_category_id()
        
    def build_category_id(self, ):
        val = self.label.unique().tolist()
        ind = list(range(0, len(val)))
        mapping = {j: i for i, j in zip(ind, val)}
        self.label = self.label.map(mapping).to_numpy()
        
    def build_edge(self):
        edge_index = np.where(self.data != 0)
        edge_index = np.vstack(edge_index).T
        return edge_index
    
    def build_train_u2i(self, edge_index):
        train_u2i = defaultdict(list)
        for d in edge_index:
            train_u2i[int(d[0])] += [int(d[1])]
        return train_u2i