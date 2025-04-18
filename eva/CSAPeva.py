#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/31 modify 2024/9/11
# @Author  : Shuheng Fang
# @Mail    : fangshuheng@gmail.com
# @File    : CSAPeva.py

from statistics import mean
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, RandomSampler
import torch
import torch.nn as nn
from get_args import get_args
import numpy as np
import networkx as nx
import time
import pickle
import random
import os
import nxmetis
from util import WeightBCEWithLogitsLoss, evaluate_prediction
from load_data import load_webkb
from tqdm import tqdm
import math
import pdb
from torch import optim
from models.GNN import GNN
from PLACE.models.QueryPrompt import QueryPromptAugmented
from models.CSAttrPrompt import CSAttrP
from QueryDatasetAttr import CNPAttributeQueryData
from tSNE import tsne_vis



class CSAPeva(nn.Module):
    def __init__(self, args, x_size, taskdata, wandb_run, device):
        super(CSAPeva, self).__init__()
        self.args = args
        self.wandb_run=wandb_run
        self.device = device
        self.input_dim = x_size
        if self.args.get_attr=='AFC' or self.args.get_attr=='AFN':
            self.num_edge_feat = 3
        else:
            self.num_edge_feat = 2
        self.criterion = WeightBCEWithLogitsLoss()
        self.initialize_prompt()
        self.initialize_model()
        self.save_path = os.path.join(self.args.model_path, self.args.dataset)
        if os.path.exists(self.save_path)==False:
            os.mkdir(self.save_path)
        self.best_epoch = 0
        #process large dataset
        if args.dataset=='reddit' or args.dataset=='product' or args.dataset=='orkut':
            raw_data_path=os.path.join(self.args.data_path, self.args.save_data_path,'raw_data.pkl')
            print("LOAD RAW DATA...")
            with open(raw_data_path, 'rb') as f:
                self.raw_data = pickle.load(f)
            self.G=self.raw_data.graph
            print("GET QUERY LIST...")
            queries_list = self.raw_data.get_communities(self.args.total_query, self.args.training_size)
            self.training_query_list=queries_list[0:self.args.training_size]
            self.test_query_list=queries_list[-self.args.test_size:]
        elif args.dataset =='cora' or args.dataset=='citeseer' or args.dataset =='cornell' or args.dataset=='texas' or args.dataset=='washington' or args.dataset=='wisconsin':
            self.taskdata = taskdata
            self.split_data(self.taskdata)


    def split_data(self, taskdata):
        print("Split training and test data")
        self.taskdata.training_test_split()
        self.training_data = taskdata.training_data
        self.test_data = taskdata.test_data
        self.valid_data = taskdata.valid_data
        print(len(self.training_data))
        print(len(self.test_data))
        print(len(self.valid_data))

    def initialize_prompt(self):
        self.prompt = QueryPromptAugmented(token_dim=self.input_dim, token_num=self.input_dim, virtual_num=self.args.virtual_num, cross_prune=0.1, inner_prune=0.5)
        self.pg_opi = optim.Adam(filter(lambda p: p.requires_grad, self.prompt.parameters()), lr=0.001, weight_decay= 0.00001)

    def initialize_model(self):
        self.cs = CSAttrP(self.args, self.input_dim, self.num_edge_feat)
        self.cs = self.cs.to(self.device)
        self.cs_opi = optim.Adam(filter(lambda p: p.requires_grad, self.cs.parameters()), lr=self.args.learning_rate, weight_decay= self.args.weight_decay) #lr=0.001,weight_decay=0.00001
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.cs_opi, 'min', factor=self.args.decay_factor, patience=self.args.decay_patience)


    def train(self):
        print("Start training model/tuning prompt...")
        f1_max = 0
        f1_max_valid = 0
        for epoch in range(self.args.total_epoch):
            pg_loss = []
            cs_loss = []
            t_epoch=0
            for i,queries in enumerate(self.training_data):
                self.cs.eval()
                self.prompt.train()
                t_qry=0
                t1=time.time()
                pg, token_num = self.prompt(queries)
                pg = pg.to(self.device)
                output,emb = self.cs(pg.x, pg.edge_index, pg.edge_attr, queries.query, token_num)
                t2=time.time()
                t_epoch=t_epoch+t2-t1
                t_qry=t_qry+t2-t1
                loss = self.criterion(output, queries.y.to(self.device), queries.mask.to(self.device))
                pg_loss.append(loss.item())
                self.pg_opi.zero_grad()
                loss.backward()
                self.pg_opi.step()

                self.cs.train()
                self.prompt.eval()
                t_qry=0
                t1=time.time()
                pg, token_num = self.prompt(queries)
                pg = pg.to(self.device)
                output,emb = self.cs(pg.x, pg.edge_index, pg.edge_attr, queries.query, token_num)
                t2=time.time()
                t_epoch=t_epoch+t2-t1
                t_qry=t_qry+t2-t1
                loss = self.criterion(output, queries.y.to(self.device), queries.mask.to(self.device))
                cs_loss.append(loss.item())
                self.cs_opi.zero_grad()
                loss.backward()
                self.cs_opi.step()

            mean_loss = mean(pg_loss)
            print("Epoch:{}/{} Pg Loss: {}".format(epoch, self.args.total_epoch, mean_loss))
            self.wandb_run.log({'prompt_tuning_loss': mean_loss},step=epoch)
            mean_loss = mean(cs_loss)
            self.scheduler.step(mean_loss)
            print("Epoch:{}/{}Training Loss: {}".format(epoch, self.args.total_epoch, mean_loss))
            self.wandb_run.log({'training_loss': mean_loss},step=epoch)
            f1_valid=self.valid()
            if f1_max_valid<f1_valid:
                f1_max_valid=f1_valid
                #save the self.cs model and prompt
                torch.save(self.prompt.state_dict(),os.path.join(self.save_path,'prompt_{}{}_{}.pt'.format(self.args.gnn_type,self.args.num_layers,self.args.get_attr)))
                torch.save(self.cs.state_dict(),os.path.join(self.save_path,'bestmodel_{}{}_{}.pt'.format(self.args.gnn_type,self.args.num_layers,self.args.get_attr)))
                self.best_epoch_valid = epoch


    def trainwithsplit(self):
        f1_max = 0
        path_sub=os.path.join(self.args.data_path, self.args.save_data_path,'subgraph_list')
        numsubgraph=len(os.listdir(path_sub))

        for epoch in tqdm(range(self.args.total_epoch)):
            t_epoch=0
            pg_loss = []
            for i,training_query in tqdm(enumerate(self.training_query_list)):
                #train prompt
                self.cs.eval()
                self.prompt.train()
                idx_list=np.random.randint(0, numsubgraph, 3)
                query_data=self.raw_data.get_one_attribute_query_tensor(training_query,self.args.num_pos, self.args.num_neg, self.args.get_attr)
                qry_list=self.query_data_partition(query_data,idx_list)
                del query_data
                t_query=0
                for j,sub in enumerate(qry_list):
                    t1=time.time()
                    pg, token_num = self.prompt(sub)
                    pg = pg.to(self.device)
                    output,emb = self.cs(pg.x, pg.edge_index, pg.edge_attr, sub.query, token_num)
                    t2=time.time()
                    t_query=t_query+t2-t1
                    t_epoch=t_epoch+t2-t1
                    loss = self.criterion(output.cpu(), sub.y, sub.mask)
                    pg_loss.append(loss.item())
                    loss.backward()
                self.pg_opi.step()
                self.pg_opi.zero_grad()

                self.cs.train()
                self.prompt.eval()
                cs_loss = []
                for j,sub in enumerate(qry_list):
                    t1=time.time()
                    pg, token_num = self.prompt(sub)
                    pg = pg.to(self.device)
                    output,emb = self.cs(pg.x, pg.edge_index, pg.edge_attr, sub.query, token_num)
                    t2=time.time()
                    t_query=t_query+t2-t1
                    t_epoch=t_epoch+t2-t1
                    loss = self.criterion(output.cpu(), sub.y, sub.mask)
                    cs_loss.append(loss.item())
                    loss.backward()
                del qry_list
                self.cs_opi.step()
                self.cs_opi.zero_grad()

            mean_loss = mean(pg_loss)
            print("Epoch:{}/{} Pg Loss: {}".format(epoch, self.args.total_epoch, mean_loss))
            self.wandb_run.log({'prompt_tuning_loss': mean_loss},step=epoch)
            mean_loss = mean(cs_loss)
            self.scheduler.step(mean_loss)
            print("Epoch:{}/{} Training Loss: {}".format(epoch, self.args.total_epoch, mean_loss))
            self.wandb_run.log({'training_loss': mean_loss},step=epoch)


    def valid(self):
        all_preds=[]
        all_targets=[]
        self.cs.eval()
        self.prompt.eval()
        for i,queries in enumerate(self.valid_data):
            pg, token_num = self.prompt(queries)
            pg = pg.to(self.device)
            output,emb = self.cs(pg.x, pg.edge_index, pg.edge_attr, queries.query, token_num)
            pred = torch.sigmoid(output)
            pred = torch.where(pred > 0.5, 1.0, 0.0)
            pred, targets = pred.view(-1), queries.y.view(-1)
            pred, targets = pred.cpu().detach().numpy(), targets.detach().numpy()
            all_preds.append(pred)
            all_targets.append(targets)
        all_preds=np.hstack(all_preds)
        all_targets=np.hstack(all_targets)
        acc, precision, recall, f1 = evaluate_prediction(all_preds, all_targets)
        print("valid: Acc={:.4f}\tPre={:.4f}\tRecall={:.4f}\tF1={:.4f}".format(acc, precision, recall, f1))
        return f1


    def eval(self):
        all_preds=[]
        all_targets=[]
        print("Start test...")
        print("Load the best model...from {}".format(self.best_epoch_valid))
        self.cs.load_state_dict(torch.load(os.path.join(self.save_path,'bestmodel_{}{}_{}.pt'.format(self.args.gnn_type,self.args.num_layers,self.args.get_attr))))
        print("Load the prompt...from{}".format(self.best_epoch_valid))
        self.prompt.load_state_dict(torch.load(os.path.join(self.save_path,'prompt_{}{}_{}.pt'.format(self.args.gnn_type,self.args.num_layers,self.args.get_attr))))
        self.cs.eval()
        self.prompt.eval()
        for i,queries in enumerate(self.test_data):
            pg, token_num = self.prompt(queries)
            pg = pg.to(self.device)
            output,emb = self.cs(pg.x, pg.edge_index, pg.edge_attr, queries.query, token_num)
            pred = torch.sigmoid(output)
            pred = torch.where(pred > 0.5, 1.0, 0.0)
            pred, targets = pred.view(-1), queries.y.view(-1)
            pred, targets = pred.cpu().detach().numpy(), targets.detach().numpy()
            all_preds.append(pred)
            all_targets.append(targets)
        all_preds=np.hstack(all_preds)
        all_targets=np.hstack(all_targets)
        acc, precision, recall, f1 = evaluate_prediction(all_preds, all_targets)
        print("Test: Acc={:.2f}\tPre={:.2f}\tRecall={:.2f}\tF1={:.2f}".format(acc*100, precision*100, recall*100, f1*100))
        return acc, precision, recall, f1


    def evalwithsplit(self):
        all_preds_=[]
        all_targets_=[]
        self.cs.eval()
        self.prompt.eval()
        path_sub=os.path.join(self.args.data_path, self.args.save_data_path,'subgraph_list')
        numsubgraph=len(os.listdir(path_sub))
        batchsize= 128
        print("Start test...")
        test_query_list=self.test_query_list
        for test_query in test_query_list:
            query_data=self.raw_data.get_one_attribute_query_tensor_for_test(test_query,self.args.num_pos, self.args.num_neg, self.args.get_attr)
            all_preds=[]
            all_targets=[]
            for i in range(numsubgraph//batchsize+1):
                if i==numsubgraph//batchsize:
                    idx_list=range(i*batchsize,numsubgraph)
                else:
                    idx_list=range(i*batchsize,(i+1)*batchsize)
                qry_list=self.query_data_partition(query_data,idx_list)
                for j,sub in enumerate(qry_list):
                    pg, token_num = self.prompt(sub)
                    pg = pg.to(self.device)
                    output,emb = self.cs(pg.x, pg.edge_index, pg.edge_attr, sub.query, token_num)
                    pred = torch.sigmoid(output)
                    pred = torch.where(pred > 0.5, 1.0, 0.0)
                    pred_ = pred.view(-1).cpu().detach().numpy()
                    targets_ = sub.y.view(-1).cpu().detach().numpy()
                    all_preds_.append(pred_)
                    all_targets_.append(targets_)
                    all_preds.append(pred_)
                    all_targets.append(targets_)
                    acc, precision, recall, f1 = evaluate_prediction(pred_, targets_)
                del qry_list
            del query_data
            all_preds=np.hstack(all_preds)
            all_targets=np.hstack(all_targets)
            acc, precision, recall, f1 = evaluate_prediction(all_preds, all_targets)

        all_preds_=np.hstack(all_preds_)
        all_targets_=np.hstack(all_targets_)
        acc, precision, recall, f1 = evaluate_prediction(all_preds_, all_targets_)
        print("The_Final_Test: Acc={:.4f}\tPre={:.4f}\tRecall={:.4f}\tF1={:.4f}".format(acc, precision, recall, f1))
        return acc, precision, recall, f1

    def query_data_partition(self,querydata,idx_list):
        #querydata
        x, x_attr, attr_index, pos, neg = querydata.x, querydata.x_attr, querydata.attr_index, querydata.pos, querydata.neg
        query = querydata.query.tolist()
        visited=[]
        path_sub=os.path.join(self.args.data_path, self.args.save_data_path,'subgraph_list')
        query_data_list=[]
        for file_name in os.listdir(path_sub):
            idx=int(file_name.strip().split('_')[-1].split('.')[0])
            if idx not in idx_list:
                continue
            file=os.path.join(self.args.data_path, self.args.save_data_path,'subgraph_list',file_name)
            with open(file, 'rb') as f:
                data_loaded = pickle.load(f)
                old_node_list = data_loaded['nodes']
            old_node_list=old_node_list+query
            newg=self.G.subgraph(old_node_list)

            idxold2new={}
            cnt=0
            for oldid in old_node_list:
                idxold2new[oldid]=cnt
                cnt=cnt+1
            remain_nodes = list(set(self.G.nodes()).difference(old_node_list))
            mapping = {n:idxold2new[n] for i, n in enumerate(old_node_list)}
            newg = nx.relabel_nodes(newg, mapping)
            new_edge_index = torch.tensor(list(newg.edges()), dtype=torch.int64).t().contiguous()

            new_x = x[torch.tensor(old_node_list)]

            new_y = querydata.y[torch.tensor(old_node_list)]
            if len(remain_nodes)!=0:
                remain_y = querydata.y[torch.tensor(list(remain_nodes))]
            else:
                remain_y = []
            new_query = [idxold2new[old] for i,old in enumerate(query)]
            new_query = torch.LongTensor(new_query)
            query_index = torch.zeros_like(new_query, dtype=torch.long)

            new_pos =list(set(pos).intersection(old_node_list))

            if len(new_pos)!=0:
                new_pos = [idxold2new[old] for i,old in enumerate(new_pos)]
            new_neg = list(set(neg).intersection(old_node_list))
            new_neg = [idxold2new[old] for i,old in enumerate(new_neg)]
            if len(new_pos)!=0:
                if self.args.num_neg<=1:#ratio
                    masked_pos = random.sample(new_pos, k = min(math.ceil(self.args.num_pos*(len(new_pos)+len(new_neg))), len(new_pos)))
                    masked_neg = random.sample(new_neg, k = min(math.ceil(self.args.num_neg*(len(new_pos)+len(new_neg))), len(new_neg)))
                else:#number
                    masked_pos = random.sample(new_pos, k = min(math.ceil(self.args.num_pos), len(new_pos)))
                    masked_neg = random.sample(new_neg, k = min(math.ceil(self.args.num_neg), len(new_neg)))
            else:
                if self.args.num_neg<=1:#ratio
                    masked_pos = []
                    masked_neg = random.sample(new_neg, k = min(math.ceil(self.args.num_neg*(len(new_pos)+len(new_neg))), len(new_neg)))
                else:#number
                    masked_pos = []
                    masked_neg = random.sample(new_neg, k = min(math.ceil(self.args.num_neg), len(new_neg)))

            if len(x_attr)==0:
                new_x_attr=[]
                new_attr_index=[]
                query_data = CNPAttributeQueryData(x=new_x, edge_index=new_edge_index, y=new_y, query=new_query,
                                           pos=masked_pos, neg=masked_neg, bias={}, query_index= query_index,
                                           x_attr=x_attr, attr_index=attr_index, remain_y=remain_y)
            else:
                new_x_attr = x_attr[torch.tensor(old_node_list)]
                query_data = CNPAttributeQueryData(x=new_x, edge_index=new_edge_index, y=new_y, query=new_query,
                                           pos=masked_pos, neg=masked_neg, bias={}, query_index= query_index,
                                           x_attr=new_x_attr, attr_index=attr_index, remain_y=remain_y)
            query_data_list.append(query_data)
        return query_data_list

