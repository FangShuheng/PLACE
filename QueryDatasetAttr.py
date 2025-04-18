from torch_geometric.data import Data, Dataset, DataLoader
import networkx as nx
from multiprocessing import Pool
import random
import torch
import os.path
import numpy as np
import math
import pickle
import nxmetis
from tqdm import tqdm
import time
import pdb
torch.multiprocessing.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2.T)
    norm_a = np.linalg.norm(vec1,axis=1,keepdims=True)
    norm_b = np.linalg.norm(vec2,axis=1,keepdims=True)
    if norm_a.any() == 0 or norm_b.any() == 0:
        return dot_product
    return dot_product / (norm_a * norm_b.T)

class CNPQueryData(Data):
    def __init__(self, x, edge_index, y, query=None, pos = None, neg = None, edge_attr = None, bias = None, query_index = None):
        super(CNPQueryData, self).__init__()
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.bias = bias
        self.y = y
        self.query = query
        self.query_index = query_index
        self.pos = pos
        self.neg = neg
        self.mask = torch.zeros(self.y.shape)  # used as the loss weight
        self.mask[pos] = 1.0
        self.mask[neg] = 1.0

    def __inc__(self, key, value):
        if key == "query":
            return 0
        elif key == "query_index":
            return 1
        else:
            return super().__inc__(key, value)


class CNPAttributeQueryData(CNPQueryData):
    def __init__(self, x, edge_index, y, query=None, pos = None, neg = None, edge_attr = None, bias = None, query_index = None,
                 x_attr = None, attr_index = None, attr_mask = None, attr_feats = None, remain_y=None):
        super(CNPAttributeQueryData, self).__init__(x, edge_index, y, query, pos, neg, edge_attr, bias, query_index)
        self.x_attr = x_attr
        self.attr_mask = attr_mask
        self.attr_feats=attr_feats
        self.remain_y=remain_y
        self.attr_index = attr_index

    def __inc__(self, key, value):
        if key == "x_attr":
            return 0
        else:
            return super().__inc__(key, value)

class TaskDataTT(object):
    def __init__(self, all_queries_data, size):
        self.data  = all_queries_data
        self.size = size


class TaskData(object):
    def __init__(self, all_queries_data, training_size, test_size):
        self.all_queries_data  = all_queries_data
        self.training_size = training_size
        self.test_size =  test_size

    def training_test_split(self):
        self.training_data, self.test_data, self.valid_data = self.all_queries_data[: self.training_size], self.all_queries_data[self.training_size:self.training_size+self.test_size],self.all_queries_data[self.training_size+self.test_size:]

    def split_on_qry(self):
        total_len = len(self.all_queries_data)
        num_subg = total_len//(self.training_size+self.test_size+1)
        self.training_data=self.all_queries_data[: self.training_size*num_subg]
        self.test_data=self.all_queries_data[self.training_size*num_subg:self.training_size*num_subg+self.test_size*num_subg]
        self.valid_data=self.all_queries_data[self.training_size*num_subg+self.test_size*num_subg:]

    def get_support_batch(self):
        support_loader = DataLoader(self.training_data, batch_size= self.num_training, shuffle=True)  # already shuffled
        return next(iter(support_loader))

    def get_query_batch(self):
        query_loader = DataLoader(self.test_data, batch_size=self.num_test, shuffle=True)
        return next(iter(query_loader))



class AttrGraphWithCommunity(object):
    def __init__(self, args, graph, communities, feats,
                 min_community_size: int = 8,
                 max_community_size: int = 10000,
                 attr_frequency: int = 3):
        self.args = args
        self.num_workers = 20
        self.graph = graph # networkx graph
        self.feats =feats # origin node feat, numpy
        self.communities = [community for community in communities if (len(community) > min_community_size and len(community)<max_community_size)]   #communities
        self.x_feats = torch.from_numpy(self.feats)
        del self.feats
        self.query_index_pre = dict()
        self.query_index = dict()
        num_attr = self.x_feats.size(-1)
        community_attr_freq = dict() # {community : {attr : frequency}}
        for community_id, community in enumerate(self.communities):
            for node in community:
                if node not in self.query_index_pre:
                    self.query_index_pre[node] = set(community)
                else:
                    self.query_index_pre[node] = self.query_index_pre[node].union(set(community))
                # get the attribute
                for attr in torch.nonzero(self.x_feats[node])[:, 0].tolist():
                    if community_id not in community_attr_freq:
                        community_attr_freq[community_id] = dict()
                    if attr not in community_attr_freq[community_id]:
                        community_attr_freq[community_id][attr] = 0
                    community_attr_freq[community_id][attr] += 1

        all_freq_attrs = set(range(num_attr))
        self.community_attribute_index = dict()
        # get the top 3 frequent attribute for each community as the candidate query attributes
        for community_id, attribute_freq in community_attr_freq.items():
            sorted_freq = sorted(attribute_freq.items(), key=lambda x: x[1], reverse=True) # sort the dict by value
            sorted_freq = sorted_freq[0 : min(len(sorted_freq), attr_frequency)] # top frequency list
            self.community_attribute_index[community_id] = set([attribute for (attribute, _) in sorted_freq])
            all_freq_attrs &= self.community_attribute_index[community_id]
        del community_attr_freq
        self.query_attributes = set()
        # remove the frequent attributes which appear in all the communities
        for community_id, freq_attr in self.community_attribute_index.items():
            self.community_attribute_index[community_id] = freq_attr.difference(all_freq_attrs)
            self.query_attributes |= self.community_attribute_index[community_id]
       #pdb.set_trace()
        for community_id, community in enumerate(self.communities):
            if community_id in self.community_attribute_index.keys():#if community has no attribute, filter the community
                self.communities[community_id]=self.communities[community_id]
            else:
                for i,nodeid in enumerate(self.communities[community_id]): #if community has no attribute, filter the community and delete self.query_index
                    if nodeid in self.query_index_pre:
                        del self.query_index_pre[nodeid]
        for idx,node in enumerate(self.query_index_pre): #filter the community where the community size is too large (almost entire graph)
            #if len(self.query_index_pre[node])<0.7*self.graph.number_of_nodes():
            self.query_index[node]=self.query_index_pre[node]
        del self.query_index_pre
        self.num_communities=len(self.communities)
        print("num communities: {}".format(self.num_communities))
        self.query_attributes_index = {attr: idx for (idx, attr) in enumerate(self.query_attributes)} # remapping the attributes for querying
        self.num_query_attributes = len(self.query_attributes_index)
        self.number_of_queries=len(self.query_index)
        print("num query attribute: {}".format(self.num_query_attributes))
        print("num query: {}".format(len(self.query_index)))
        self.edge_index = torch.tensor(list(self.graph.edges()), dtype=torch.int64).t().contiguous()




    def sample_one_for_multi_node_AFC(self, query, num_pos, num_neg, max_query_node: int = 3, max_query_attribute: int = 3):
        num_query_node = random.randint(1, max_query_node)
        num_query_attribute = random.randint(1, max_query_attribute)
        #query = random.sample(self.communities[community_id], k = num_query_node)#117,128
        pos = list(self.query_index[query])
        querys=random.sample(pos,k=num_query_node-1)
        querys.append(query)
        neg = list(set(range(self.graph.number_of_nodes())).difference(self.query_index[query]))
        pos = list(set(pos).difference(querys))
        if num_neg<=1:#ratio
            masked_pos = random.sample(pos, k = min(math.ceil(num_pos*(len(pos)+len(neg))), len(pos)))
            masked_neg = random.sample(neg, k = min(math.ceil(num_neg*(len(pos)+len(neg))), len(neg)))
        else:#number
            masked_pos = random.sample(pos, k = min(math.ceil(num_pos), len(pos)))
            masked_neg = random.sample(neg, k = min(math.ceil(num_neg), len(neg)))

        #sample query attribute from community
        for community_id, community in enumerate(self.communities):
            if query in self.communities[community_id]:
                if len(self.community_attribute_index[community_id])>num_query_attribute:
                    query_attributes = random.sample(self.community_attribute_index[community_id], k=num_query_attribute)#637
                else:
                    query_attributes = self.community_attribute_index[community_id]
        return querys, pos, neg, masked_pos, masked_neg, query_attributes

    def sample_one_for_multi_node_AFN(self, query, num_pos, num_neg, max_query_node: int = 3, max_query_attribute: int = 3):
        num_query_node = random.randint(1, max_query_node)
        num_query_attribute = random.randint(1, max_query_attribute)
        if query==2055:
            num_query_node=3
        #num_query_attribute = random.randint(1, max_query_attribute)
        #query = random.sample(self.communities[community_id], k = num_query_node)#117,128
        pos = list(self.query_index[query])
        querys=random.sample(pos,k=num_query_node-1)
        querys.append(query)
        neg = list(set(range(self.graph.number_of_nodes())).difference(self.query_index[query]))
        pos = list(set(pos).difference(querys))
        if num_neg<=1:#ratio
            masked_pos = random.sample(pos, k = min(math.ceil(num_pos*(len(pos)+len(neg))), len(pos)))
            masked_neg = random.sample(neg, k = min(math.ceil(num_neg*(len(pos)+len(neg))), len(neg)))
        else:#number
            masked_pos = random.sample(pos, k = min(math.ceil(num_pos), len(pos)))
            masked_neg = random.sample(neg, k = min(math.ceil(num_neg), len(neg)))
        #sample query attributes from query node
        query_attributes=[]
        for q in querys:
            for attr in torch.nonzero(self.x_feats[q])[:, 0].tolist():
                query_attributes.append(attr)
        if len(query_attributes)>num_query_attribute:
            query_attributes = random.sample(query_attributes, k=num_query_attribute)#637
        #print("num query attr:{}".format(len(query_attributes)))
        return querys, pos, neg, masked_pos, masked_neg, query_attributes

    def sample_one_for_multi_node_emA(self, query, num_pos, num_neg, max_query_node: int = 3, max_query_attribute: int = 1):
        num_query_node = random.randint(1, max_query_node)
        #num_query_attribute = random.randint(1, max_query_attribute)
        #query = random.sample(self.communities[community_id], k = num_query_node)#117,128
        pos = list(self.query_index[query])
        querys=random.sample(pos,k=num_query_node-1)
        querys.append(query)
        neg = list(set(range(self.graph.number_of_nodes())).difference(self.query_index[query]))
        pos = list(set(pos).difference(querys))
        if num_neg<=1:#ratio
            masked_pos = random.sample(pos, k = min(math.ceil(num_pos*(len(pos)+len(neg))), len(pos)))
            masked_neg = random.sample(neg, k = min(math.ceil(num_neg*(len(pos)+len(neg))), len(neg)))
        else:#number
            masked_pos = random.sample(pos, k = min(math.ceil(num_pos), len(pos)))
            masked_neg = random.sample(neg, k = min(math.ceil(num_neg), len(neg)))
        #emA means no query attributes
        query_attributes=[]
        return querys, pos, neg, masked_pos, masked_neg, query_attributes


    def get_communities(self, total_query, training_size):
        assert total_query <= len(self.query_index), "total number of queries surpass the max number of queries."
        assert total_query > training_size, "training size surpasses the total number of queries"
        queries = random.choices(list(self.query_index.keys()),k=total_query)
        return queries

    def get_one_attribute_query_tensor(self, query, num_pos, num_neg, get_attr):
        if get_attr =='AFC':
            query, pos, neg, masked_pos, masked_neg, query_attributes = self.sample_one_for_multi_node_AFC(query, num_pos, num_neg)
        elif get_attr =='AFN':
            query, pos, neg, masked_pos, masked_neg, query_attributes = self.sample_one_for_multi_node_AFN(query, num_pos, num_neg)
        elif get_attr =='emA':
            query, pos, neg, masked_pos, masked_neg, query_attributes = self.sample_one_for_multi_node_emA(query, num_pos, num_neg)

        dist = {}

        y = torch.zeros(size=(self.graph.number_of_nodes(),), dtype=torch.float)
        y[pos] = 1
        y[query] = 1

        query = torch.LongTensor(query)
        query_index = torch.zeros_like(query, dtype=torch.long)
        x=self.x_feats.to(torch.float32)

        if len(query_attributes)==0:
             x_attr=[]
             attr_index=[]
             query_data = CNPAttributeQueryData(x=x, edge_index=self.edge_index, y=y, query=query,
                                           pos=masked_pos, neg=masked_neg, bias= dist, query_index= query_index,
                                           x_attr=x_attr, attr_index=attr_index)
        else:
            x_attr = torch.zeros(size=(self.graph.number_of_nodes(), len(query_attributes))).to(torch.float32)
            attr_index = torch.zeros(size=(len(query_attributes),1)).to(torch.float32)
            for i,attribute in enumerate(query_attributes):
                x_attr[:, i] = self.x_feats[:, attribute]
                attr_index[i] = attribute
            query_data = CNPAttributeQueryData(x=x, edge_index=self.edge_index, y=y, query=query,
                                           pos=masked_pos, neg=masked_neg, bias= dist, query_index= query_index,
                                           x_attr=x_attr, attr_index=attr_index)
        return query_data


    def get_attributed_task(self, queries, num_shots, test_size, num_pos, num_neg, get_attr='AFC'):
        if self.args.model_name=='CSAttrP':
            get_tensor_func = self.get_one_attribute_query_tensor
        all_queries_data = list()
        for query in queries:
            all_queries_data.append(get_tensor_func(query, num_pos, num_neg, get_attr))
        task = TaskData(all_queries_data, training_size=num_shots, test_size=test_size)
        return task

def np_save_if_not_existed(path, saved_data):
    if not os.path.exists(path):
        saved_data_numpy = np.array([saved_data], dtype=object)
        np.save(path, saved_data_numpy)
