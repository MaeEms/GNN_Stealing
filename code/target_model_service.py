import dgl
from dgl import load_graphs
from src.gin import *
from src.gat import *
from src.sage import *
from src.utils import *
import os
import pickle
import json
import argparse
from scipy import sparse
import torch
import numpy as np
from core.model_handler import ModelHandler
from flask import Flask, url_for, jsonify, request
from flask import Flask
from flask_cors import CORS

target_model_app = Flask(__name__)

# 解决有关进程的异常
# torch.multiprocessing.set_start_method('spawn')
# 解决跨域
cors = CORS(target_model_app, resources=r'/*')

torch.set_num_threads(1)
torch.manual_seed(0)

# 参数设置
gpu = 1
if gpu >= 0:
    device = th.device('cuda:%d' % gpu)
else:
    device = th.device('cpu')
query_ratio = 1.0


"""
目标模型初始化
输入：数据集名称、特征形状、数据集类别； 目标模型名称，目标模型隐藏层节点数
输出：目标模型，目标模型模型参数
"""
def target_model_init(dataset, feature_shape, n_classes, target_model_name, target_model_dim):
    # 加载模型参数，如果目标模型不是GIN，则model_args是没有用到的
    model_args = pickle.load(open('./target_model_' + target_model_name +
                                  '_' + str(target_model_dim) + '/model_args', 'rb'))
    model_args = model_args.__dict__
    model_args["gpu"] = gpu
    # 如果目标模型是GIN，则创建一个GIN模型并加载其状态字典；否则，直接从文件中加载目标模型
    if target_model_name == 'gin':
        print('target model: ', target_model_name)
        target_model = GIN(feature_shape,
                           model_args['num_hidden'],
                           n_classes,
                           model_args['num_layers'],
                           F.relu,
                           model_args['batch_size'],
                           model_args['num_workers'],
                           model_args['dropout'])
        target_model.load_state_dict(torch.load('./target_model_' + target_model_name + '_' + str(
            target_model_dim) + '/' + 'target_model_gin_' + dataset,
                                                map_location=torch.device('cpu')))
    else:
        target_model = torch.load('./target_model_' + target_model_name + '_' + str(
            target_model_dim) + '/' + 'target_model_' + target_model_name + '_' + dataset,
                                  map_location=torch.device('cpu'))
    return target_model, model_args


"""
测试目标模型
输入：目标模型名称、目标模型、目标模型参数；测试集
输出：目标模型准确率、目标模型预测结果
"""
def evaluate_target_model(target_model_name, target_model, target_model_args, test_g):
    # 加载目标模型
    target_model = target_model.to(device)
    target_model_args["gpu"] = gpu
    if target_model_name == 'sage':
        test_acc, pred, embs, class_acc = evaluate_sage_target(target_model,
                                                    test_g,
                                                    test_g.ndata['features'],
                                                    test_g.ndata['labels'],
                                                    test_g.nodes(),
                                                    target_model_args["batch_size"],
                                                    device)
    elif target_model_name == 'gat':
        test_acc, pred, embs,class_acc = evaluate_gat_target(target_model,
                                                test_g,
                                                test_g.ndata['features'],
                                                test_g.ndata['labels'],
                                                test_g.nodes(),
                                                target_model_args['val_batch_size'],
                                                target_model_args['head'],
                                                device)
    elif target_model_name == 'gin':
        test_acc, pred, embs,class_acc = evaluate_gin_target(target_model,
                                                test_g,
                                                test_g.ndata['features'],
                                                test_g.ndata['labels'],
                                                test_g.nodes(),
                                                target_model_args["batch_size"],
                                                device)
    return test_acc, pred, embs, class_acc


dataset = "citeseer_full"
target_model_name = "gat"
target_model_dim = 256

# 加载数据集
g, n_classes = load_graphgallery_data(dataset)
feature_shape = g.ndata['features'].shape[1]

# 初始化目标模型
target_model, target_model_args = target_model_init(dataset, feature_shape, n_classes, target_model_name, target_model_dim)


@target_model_app.route('/get_query_graph_pred', methods=['POST'])
def get_query_graph_pred():
    # 获取接收到的数据
    serialized_test_g = request.data

    # 反序列化字节流以获取图对象
    test_g = pickle.loads(serialized_test_g)

    # 计算目标模型准确率
    query_acc, query_preds, query_embs, class_acc= evaluate_target_model(target_model_name, target_model, target_model_args, test_g)

    response_data = {
        'query_preds':query_preds.tolist()
    }

    return jsonify(response_data)
    

if __name__ == '__main__':
    target_model_app.run(host="0.0.0.0", port=6020)