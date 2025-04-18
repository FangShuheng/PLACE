#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Shuheng Fang
# @Mail    : fangshuheng@gmail.com
# @File    : util.py
# @description: util function for this project


import torch
import numpy as np
import random
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.module import Module
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score


def seed_all(seed: int =0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("set all seed!")

class WeightBCEWithLogitsLoss(Module):
	"""
	BCEwithLogitsLoss with element-wise weight
	"""
	def __init__(self):
		super(WeightBCEWithLogitsLoss, self).__init__()
		self.bce = BCEWithLogitsLoss(reduction="none")

	def forward(self, inputs, target, weights=None):
		loss = self.bce(inputs, target)
		if weights is not None:
			loss = torch.mul(loss, weights)
		loss = torch.sum(loss)
		return loss

def evaluate_prediction(pred, targets):
    acc = accuracy_score(targets, pred)
    precision = precision_score(targets, pred)
    recall = recall_score(targets, pred)
    f1 = f1_score(targets, pred)
    return acc, precision, recall, f1



def get_act_layer(act_type: str):
	if act_type == "relu":
		return nn.ReLU()
	elif act_type == "tanh":
		return nn.Tanh()
	elif act_type == "leaky_relu":
		return nn.LeakyReLU()
	elif act_type == "prelu":
		return nn.PReLU()
	elif act_type == 'grelu':
		return nn.GELU()
	elif act_type == "none":
		return lambda x : x
	else:
		raise NotImplementedError("Error: %s activation function is not supported now." % (act_type))



def jaccard_similarity(set1, set2):
    """计算两个集合的 Jaccard 相似度"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def dice_coefficient(set1, set2):
    """计算两个集合的 Dice 系数"""
    intersection = len(set1.intersection(set2))
    return 2 * intersection / (len(set1) + len(set2))
