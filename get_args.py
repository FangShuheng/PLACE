#!/usr/bin/env python
# @Author  : Shuheng Fang
# @Mail    : fangshuheng@gmail.com
# @File    : get_args.py
# @description: parameter configuration


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch

def get_args():
    parser = ArgumentParser("PromCS", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")

    # set the dataset
    parser.add_argument('--dataset', type=str, default='washington', help='Dataset.') #transfer_facebook
    parser.add_argument('--data_path', type=str, default='/home/shfang/PromCS/data', help='data path.')
    parser.add_argument('--save_data_path', type=str, default='orkut_split_graph', help='save data path.')
    parser.add_argument('--model_path', type=str, default='/home/shfang/PromCS/savedmodel', help='data path.')

    #GNN
    parser.add_argument('--total_epoch', default=200, type=int, help='model training total epoch')
    parser.add_argument('--tune_epoch', default=200, type=int, help='prompt tune epoch')
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--decay_factor', type=float, default=0.8, help='decay rate of (gamma).')
    parser.add_argument('--decay_patience', type=int, default=10, help='num of epochs for one lr decay.')

    # model config
    parser.add_argument('--model_name', default='CSAttrP', type=str, help='model name') #CSAttrPtransfer
    #GNN
    parser.add_argument("--gnn_type", default="RGCN", type=str,help="GNN type") # GCN, GAT, GATBias, RGCN,GIN?
    parser.add_argument("--num_g_hid", default=128, type=int, help="hidden dim for transforming nodes")
    parser.add_argument("--num_e_hid", default=128, type=int, help="hidden dim for transforming edges") #for NN, NNGIN, NNGINConcat, not use now
    parser.add_argument('--num_layers', default=3, type=int, help='number of layers')
    parser.add_argument("--gnn_out_dim", default=128, type=int, help="number of output dimension") #128
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument("--act_type", default="relu", type=str, help="activation layer function for MLP and between GNN layers")
    parser.add_argument("--gnn_act_type", default="relu", type=str, help="activation layer inside gnn aggregate/combine function")

    # community search config
    parser.add_argument('--virtual_num', default=1, type=int, help='total number of virtual nodes.')
    parser.add_argument('--total_query', default=150, type=int, help='total number of queries: training,test,validation.')#170
    parser.add_argument('--training_size', default=50, type=int, help='number of training queries')#150
    parser.add_argument('--test_size', default=50, type=int, help='number of test queries')#20
    parser.add_argument('--num_pos', default=1, type=float, help='number of positive samples for each query')
    parser.add_argument('--num_neg', default=1, type=float, help='number of negative samples for each query')
    parser.add_argument('--max_samplenode', default=5000, type=int, help='the max number of nodes for layer-wise sampling')
    parser.add_argument('--cut_samplenode', default=1000, type=int, help='the number of nodes to select whether use attribute similarity filtering')
    parser.add_argument('--sim_thr', default=0.3, type=float, help='similar threshold')
    parser.add_argument('--get_attr', default='AFC', type=str, help='AFC,AFN,emA')
    # set the hardware parameter
    parser.add_argument('--n', default=0, type=int, help='GPU ID')

    #other setting
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()



    return args
