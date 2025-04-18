#!/usr/bin/env python
# @Author  : Shuheng Fang
# @Mail    : fangshuheng@gmail.com
# @File    : main.py

import torch
from get_args import get_args
import threading
import wandb
import numpy as np
import time
import random
import os
from util import seed_all
from load_data import load_data
from eva.CSAPeva import CSAPeva
import wandb


def main(args):
    seed_all(args.seed)
    wandb_run=wandb.init(config=args,project='PromCS',dir='/home/shfang/PromCS/wandb/',job_type="training",name="fe01{}_{}{}_{}".format(args.dataset,args.gnn_type,args.num_layers,args.get_attr),reinit=True)
    device = torch.device('cuda:{}'.format(args.n) if torch.cuda.is_available() else 'cpu')

    if args.model_name == 'CSAttrP':
        #small dataset
        if  args.dataset =='cora' or args.dataset=='citeseer' or args.dataset =='cornell' or args.dataset=='texas' or args.dataset=='washington' or args.dataset=='wisconsin':
            taskdata, x_size = load_data(args)
            csap = CSAPeva(args, x_size, taskdata, wandb_run, device)
            csap.train()
            csap.eval()
        #large dataset
        elif args.dataset=='reddit' or args.dataset=='product' or args.dataset=='orkut':
            x_size = load_data(args)
            taskdata=[]
            csap = CSAPeva(args, x_size, taskdata, wandb_run, device)
            csap.trainwithsplit()




if __name__ == "__main__":
    args = get_args()
    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    try:
        pid = os.getpid()
    except:
        print("Error encountered, pid is set to 0")
        pid=0
    run_event = threading.Event()
    run_event.clear()
    main(args)
