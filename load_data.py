#!/usr/bin/env python
# @Author  : Shuheng Fang
# @Mail    : fangshuheng@gmail.com
# @File    : load_data.py
# @Describe: dataloader

import nxmetis
import time
from tqdm import tqdm
import numpy as np
import os
import pickle
import torch
from scipy.sparse import coo_matrix
from scipy import io as sio
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, Dataset
from torch_geometric.datasets import DBLP
from torch_geometric.datasets import Reddit
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from QueryDataset import RawGraphWithCommunity
from QueryDatasetHomo import HomoGraphWithCommunity
from QueryDatasetAttr import AttrGraphWithCommunity, TaskData, TaskDataTT
from util import text2map, build_edge
import networkx as nx
import sys
import pickle as pkl
import torch_geometric

#dfs sample
def dfs(graph, start, nodeset, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    if start in nodeset:
        nodeset.remove(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, nodeset, visited)

def load_citation_graph(args):
    feature = np.load('/home/shfang/PromCS/data/' + args.dataset + '/features.npy', allow_pickle=True)
    with open('/home/shfang/PromCS/data/'+args.dataset+'/ind.'+args.dataset+'.graph','rb') as f:
        if sys.version_info > (3, 0):
            graph = pkl.load(f, encoding='latin1')
    with open('/home/shfang/PromCS/data/' + args.dataset + '/label.pkl', 'rb') as f:
        info = pkl.load(f)
    edge_list = []
    #load data
    for i,node1 in enumerate(graph):
        for j, node2 in enumerate(graph[i]):
            edge_list.append((node1, node2))
    g = nx.Graph()
    g.add_edges_from(edge_list)

    #find connected graph component
    node_set = set(g.nodes())
    visited = dict()
    partnum = 1
    maxpart = 0
    while len(node_set)!=0 :
        visited[partnum]=set()
        dfs(graph,node_set.pop(),node_set, visited[partnum])
        if maxpart< len(visited[partnum]):
            maxpart=len(visited[partnum])
            partid=partnum
        partnum = partnum+1

    # extract the largest connected component of the graph and reindex the node
    idxold2new={}
    cnt=0
    old_node_list = list(visited[partid])
    for oldid in old_node_list:
        idxold2new[oldid]=cnt
        cnt=cnt+1
    sub = g.subgraph(old_node_list)
    mapping = {n: idxold2new[n] for i, n in enumerate(sub.nodes())}
    sub_new = nx.relabel_nodes(sub, mapping)
    print(nx.is_connected(sub_new))
    print(sub_new.number_of_nodes(), sub_new.number_of_edges())

    #communities
    communities = list()
    label_dict = {}
    for idx, node in enumerate(old_node_list):
        if info[str('0' + '_' + str(node))][0] in label_dict.keys():
            label_dict[info[str('0' + '_' + str(node))][0]].append(idxold2new[node])
        else:
            label_dict[info[str('0' + '_' + str(node))][0]] = [idxold2new[node]]
    for idx, node_ls in enumerate(label_dict):
        communities.append(label_dict[node_ls])

    #feats
    feats = np.vstack(([feature[np.array(x)] for j, x in enumerate(old_node_list)]))

    raw_data = AttrGraphWithCommunity(args, sub_new, communities, feats, min_community_size=8)
    x_size = feats.shape[1]
    return raw_data, x_size


def load_webkb(args):
    data_dir='/home/shfang/PromCS/data/webkb/'
    feat = np.genfromtxt("{}{}.content".format(data_dir, args.dataset), dtype=np.dtype(str))
    node_name = feat[:, 0]
    node_map = {}
    i = 0
    for j in node_name:
        node_map[j] = i
        i = i + 1
    feature = feat[:, 1:-1].astype(np.int32)
    label=feat[:,-1]


    edge = open("{}{}.cites".format(data_dir, args.dataset), "r")
    readedge = edge.readlines()
    edge_list = []
    for string in readedge:
        string = string.strip().split(' ')
        edge_list.append([node_map[string[0]], node_map[string[1]]])
    edge.close()
    g = nx.Graph()
    g.add_edges_from(edge_list)
    print(g.number_of_nodes(), g.number_of_edges())
    print(nx.is_connected(g)) #False

    #find connected graph component
    node_set = set(g.nodes())
    graph=dict()
    for node in node_set:
        graph[node]=nx.neighbors(g, node)
    visited = dict()
    partnum = 1
    maxpart = 0
    while len(node_set)!=0 :
        visited[partnum]=set()
        dfs(graph, node_set.pop(),node_set, visited[partnum])
        if maxpart< len(visited[partnum]):
            maxpart=len(visited[partnum])
            partid=partnum
        partnum = partnum+1

    # extract the largest connected component of the graph and reindex the node
    idxold2new={}
    cnt=0
    old_node_list = list(visited[partid])
    for oldid in old_node_list:
        idxold2new[oldid]=cnt
        cnt=cnt+1
    sub = g.subgraph(old_node_list)
    mapping = {n: idxold2new[n] for i, n in enumerate(sub.nodes())}
    sub_new = nx.relabel_nodes(sub, mapping)
    print(nx.is_connected(sub_new))
    print(sub_new.number_of_nodes(), sub_new.number_of_edges())

    # communities
    communities = list()
    label_dict = {}
    for idx, node in enumerate(old_node_list):
        if label[node] in label_dict.keys():
            label_dict[label[node]].append(idxold2new[node])
        else:
            label_dict[label[node]] = [idxold2new[node]]
    for idx, node_ls in enumerate(label_dict):
        communities.append(label_dict[node_ls])

    #feats
    feats = np.vstack(([feature[np.array(x)] for j, x in enumerate(old_node_list)]))

    raw_data = AttrGraphWithCommunity(args, sub_new, communities, feats, min_community_size=8)
    x_size = feats.shape[1]
    return raw_data, x_size



def load_reddit(args):
    path = "/home/shfang/PromCS/data/reddit"
    feature=torch.load("/home/shfang/data/reddit/feat.pt")
    feature=feature.numpy()
    save_path=os.path.join(args.data_path, args.save_data_path)
    if not os.path.exists(path):
        os.makedirs(path)
    dataset = Reddit(path)
    data = dataset[0]
    num_feat = data.x.shape[1] #602

    edge_list=data.edge_index.t().tolist()
    graph = nx.Graph()
    graph.add_nodes_from(list(range(data.num_nodes)))
    graph.add_edges_from(edge_list)
    print(graph.number_of_nodes(), graph.number_of_edges())

    glob_communities = dict()
    communities = list()
    for node_id, label in enumerate(data.y.numpy().tolist()):
        if label not in glob_communities.keys():
            glob_communities[label] = list()
        glob_communities[label].append(node_id)
    for node_id, label in enumerate(glob_communities):
        communities.append(glob_communities[label])

    if not os.path.exists(os.path.join(save_path,'subgraph_list')):
        os.makedirs(os.path.join(save_path,'subgraph_list'))
        print("BEGIN PARTITION...")
        t=time.time()
        #partition the large graph
        obj,subgraph_nodes=nxmetis.partition(graph,graph.number_of_nodes()//1000)
        print("partition time: {}".format(time.time()-t))
        for idx, nodes in enumerate(tqdm(subgraph_nodes)):
            #save node list for shard
            subgraph_path=os.path.join(save_path,'subgraph_list','subgraph_{}.pkl'.format(idx))
            with open(subgraph_path, 'wb') as f:
                data_to_save = {
                    'nodes': nodes
                }
                #using pickle to save
                pickle.dump(data_to_save, f)
        print("Subgraph nodes saved to files.")
        del subgraph_nodes

    raw_data = AttrGraphWithCommunity(args, graph, communities, feature, min_community_size=8)
    raw_data_path=os.path.join(save_path,'raw_data.pkl')
    with open(raw_data_path, 'wb') as f:
        pickle.dump(raw_data, f)
    del raw_data



def load_product(args):
    path = "/home/shfang/data/product"
    save_path=os.path.join(args.data_path, args.save_data_path)
    feature=torch.load("/home/shfang/data/product/feat.pt")
    feature=feature.numpy()
    if not os.path.exists(path):
        os.makedirs(path)
    dataset = PygNodePropPredDataset(name='ogbn-products', root='/home/shfang/data/product/')
    data = dataset[0]
    graph=torch_geometric.utils.to_networkx(data,to_undirected=True)
    print(graph.number_of_nodes(), graph.number_of_edges())

    glob_communities = dict()
    communities = list()
    for node_id, label in enumerate(data.y.numpy().tolist()):
        label=int(label[0])
        if label not in glob_communities.keys():
            glob_communities[label] = list()
        glob_communities[label].append(node_id)
    for node_id, label in enumerate(glob_communities):
        communities.append(glob_communities[label])
    del data

    if not os.path.exists(os.path.join(save_path,'subgraph_list')):
        os.makedirs(os.path.join(save_path,'subgraph_list'))
        print("BEGIN PARTITION...")
        t=time.time()
        obj,subgraph_nodes=nxmetis.partition(graph,graph.number_of_nodes()//1000)
        print("partition time: {}".format(time.time()-t))
        for idx, nodes in enumerate(tqdm(subgraph_nodes)):
            subgraph_path=os.path.join(save_path,'subgraph_list','subgraph_{}.pkl'.format(idx))
            with open(subgraph_path, 'wb') as f:
                data_to_save = {
                    'nodes': nodes
                }
                pickle.dump(data_to_save, f)
        print("Subgraph nodes saved to files.")
        del subgraph_nodes,edge_index

    raw_data = AttrGraphWithCommunity(args, graph, communities, feature, min_community_size=8)
    raw_data_path=os.path.join(save_path,'raw_data.pkl')
    with open(raw_data_path, 'wb') as f:
        pickle.dump(raw_data, f)
    del raw_data


def load_orkut(args):
    edges_file="/home/shfang/PromCS/data/orkut/com-orkut.ungraph.txt"
    communities_file="/home/shfang/PromCS/data/orkut/com-orkut.top5000.cmty.txt"
    feature=torch.load("/home/shfang/PromCS/data/orkut/feat.pt")
    feature=feature.numpy()
    save_path=os.path.join(args.data_path, args.save_data_path)

    graph = nx.Graph()
    with open(edges_file,'r') as f:
        next(f)
        next(f)
        next(f)
        next(f)
        for line in f:
            from_node,to_node=map(int,line.split())
            graph.add_edge(from_node,to_node)
    print(graph.number_of_nodes(), graph.number_of_edges())
    original_nodes = list(graph.nodes())
    node_mapping = {original_node: new_index for new_index, original_node in enumerate(original_nodes)}
    graph = nx.relabel_nodes(graph, node_mapping)

    communities = []
    with open(communities_file, 'r') as f:
        for line in f:
            community = list(map(int, line.split()))
            community = [node_mapping[node] for node in community]
            communities.append(community)

    if not os.path.exists(os.path.join(save_path,'subgraph_list')):
        os.makedirs(os.path.join(save_path,'subgraph_list'))
        print("BEGIN PARTITION...")
        t=time.time()
        obj,subgraph_nodes=nxmetis.partition(graph,graph.number_of_nodes()//1000)
        print("partition time: {}".format(time.time()-t))
        for idx, nodes in enumerate(tqdm(subgraph_nodes)):
            subgraph_path=os.path.join(save_path,'subgraph_list','subgraph_{}.pkl'.format(idx))
            with open(subgraph_path, 'wb') as f:
                data_to_save = {
                    'nodes': nodes
                }
                pickle.dump(data_to_save, f)
        print("Subgraph nodes saved to files.")
        del subgraph_nodes

    raw_data = AttrGraphWithCommunity(args, graph, communities, feature, min_community_size=8)
    raw_data_path=os.path.join(save_path,'raw_data.pkl')
    with open(raw_data_path, 'wb') as f:
        pickle.dump(raw_data, f)
    del raw_data



def load_data(args, remove_self_loop=False):
    if args.dataset =='cora' or args.dataset=='citeseer':
        raw_data, x_size =load_citation_graph(args)
        queries_list = raw_data.get_communities(args.total_query, args.training_size)
        task = raw_data.get_attributed_task(queries_list, args.training_size, args.test_size, args.num_pos, args.num_neg, args.get_attr)
        return task, x_size

    elif args.dataset =='cornell' or args.dataset=='texas' or args.dataset=='washington' or args.dataset=='wisconsin':
        raw_data, x_size =load_webkb(args)
        queries_list = raw_data.get_communities(args.total_query, args.training_size)
        task = raw_data.get_attributed_task(queries_list, args.training_size, args.test_size, args.num_pos, args.num_neg, args.get_attr)
        return task, x_size

    elif args.dataset=='reddit' or args.dataset=='product' or args.dataset=='orkut':
        if not os.path.exists(os.path.join(args.data_path, args.save_data_path)):
            print("LOAD ORIGINAL DATA...")
            os.makedirs(os.path.join(args.data_path, args.save_data_path))
            if args.dataset=='reddit':
                load_reddit(args)
            elif args.dataset=='product':
                load_product(args)
            elif args.dataset=='orkut':
                load_orkut(args)
        if args.dataset=='reddit':
            x_size=1164
        elif args.dataset=='product':
            x_size = 12245
        elif args.dataset=='orkut':
            x_size = 15362
        return x_size
    else:
        raise NotImplementedError('Unsupported dataset {}'.format(args.dataset))

