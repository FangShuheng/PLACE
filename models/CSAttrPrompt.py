import torch.nn as nn
import torch
import torch.nn.functional as F
from models.GNN import GNN




class CSAttrP(nn.Module):
    # community search model using prompt
    def __init__(self, args, input_dim, num_edge_feat):
        super(CSAttrP, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.num_edge_feat = num_edge_feat
        self.gnn = GNN(args, self.input_dim, self.num_edge_feat)


    def forward(self, x, edge_index, edge_attr, query, token_num):
        x_hid = self.gnn(x, edge_index, edge_attr)
        x_hid_ = x_hid[token_num:,:]
        q = x_hid_[query]
        hid = torch.einsum("nc,kc->nk", [q, x_hid_])
        prob = torch.mean(hid,dim=0)
        return prob,x_hid



