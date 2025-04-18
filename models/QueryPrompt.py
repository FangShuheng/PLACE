
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from deprecated.sphinx import deprecated
from sklearn.cluster import KMeans


class QueryPrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_group, group_num=1, inner_prune=None):
        super(QueryPrompt, self).__init__()
        self.device='cpu'
        self.inner_prune = inner_prune
        self.token_list = torch.nn.Parameter(torch.empty(token_num_per_group, token_dim))
        self.vir_list = torch.nn.Parameter(torch.empty(group_num, token_dim))
        self.token_init()

    def token_init(self):
        torch.nn.init.kaiming_uniform_(self.token_list, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        torch.nn.init.kaiming_uniform_(self.vir_list, nonlinearity='leaky_relu', mode='fan_in', a=0.01)

    def inner_structure_update(self, attr_index):
        if len(attr_index)!=0:
            x, edge_index = self.attr_view(attr_index)
        else:
            x, edge_index = self.vir_view()
        pg=Data(x=x, edge_index=edge_index)
        return pg

    def attr_view(self, attr_index):
        token_list4query = torch.zeros(((attr_index.size(0)+self.vir_list.size(0),self.token_list.size(1))))#.to(self.device)
        for i,idx in enumerate(attr_index):
            idx= int(idx.item())
            token_list4query[i,:] = self.token_list[idx,:]
        token_list4query = torch.cat([token_list4query, self.vir_list], dim=0)
        token_dot = torch.mm(token_list4query, torch.transpose(token_list4query, 0, 1))
        token_sim = torch.sigmoid(token_dot)
        inner_adj = torch.where(token_sim < self.inner_prune, 0, 1)
        edge_index = inner_adj.nonzero().t().contiguous()
        return token_list4query,edge_index

    def vir_view(self):
        token_list4query = torch.zeros(((self.vir_list.size(0),self.vir_list.size(1))))
        token_list4query = torch.cat([token_list4query, self.vir_list], dim=0)
        token_dot = torch.mm(token_list4query, torch.transpose(token_list4query, 0, 1))
        token_sim = torch.sigmoid(token_dot)
        inner_adj = torch.where(token_sim < self.inner_prune, 0, 1)
        edge_index = inner_adj.nonzero().t().contiguous()
        return token_list4query,edge_index


class QueryPromptAugmented(QueryPrompt):
    def __init__(self, token_dim, token_num, virtual_num, cross_prune=0.1, inner_prune=0.01):
        super(QueryPromptAugmented, self).__init__(token_dim, token_num, virtual_num, inner_prune)
        self.cross_prune = cross_prune
    def forward(self, queries):
        x, edge_index, x_attr, attr_index = queries.x, queries.edge_index, queries.x_attr, queries.attr_index
        query = queries.query
        pg = self.inner_structure_update(attr_index)
        if len(pg.edge_index)!=0:
            inner_edge_index = pg.edge_index
            inner_edge_type = 2*torch.ones(size=(1,inner_edge_index.size(1)),dtype=torch.int64)
        #attr num+vir_node
        token_num = pg.x.shape[0]
        #renumber edge index
        edge_index = edge_index + token_num
        edge_type=torch.zeros(size=(1,edge_index.size(1)), dtype=torch.int64)
        cei=[]
        #find nodes those has the attr
        if len(x_attr)!=0:
            for i in range(x_attr.size(1)):
                attr = x_attr[:,i]
                node = attr.nonzero()
                for j, n in enumerate(node):
                    n=int(n.item())
                    cei.append((i, n+token_num))
        #virtual node connect with query nodes
        for i, n in enumerate(query):
            for j in range(len(attr_index),token_num):
                cei.append((j, n+token_num))
        cross_edged_index=torch.ones(size=(2,len(cei)),dtype=torch.int64)
        for i, e in enumerate(cei):
            cross_edged_index[0][i],cross_edged_index[1][i] = e[0],e[1]

        cross_edge_type = torch.ones(size=(1,cross_edged_index.size(1)), dtype=torch.int64)
        if len(pg.edge_index)!=0:
            edge_index = torch.cat([edge_index, cross_edged_index,inner_edge_index], dim = 1)
            edge_type = torch.cat([edge_type, cross_edge_type,inner_edge_type], dim = 1)
        else: #for emA
            edge_index = torch.cat([edge_index, cross_edged_index], dim = 1)
            edge_type = torch.cat([edge_type, cross_edge_type], dim = 1)
        x = torch.cat([pg.x, x], dim=0)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_type.squeeze())
        return data,token_num











