from src.utils import delete_dgl_graph_edge
from src.gin import *
from src.gat import *
from src.sage import *
from src.sagesurrogate import *
from src.ginsurrogate import *
from src.gatsurrogate import *
from src.utils import *
from src.constants import *
from scipy import sparse
import os
import pickle
import argparse
import torch
import numpy as np
from core.model_handler import ModelHandler
from collections import defaultdict
import requests

"""
加载数据集
输入：数据集名称
输出：特征形状、数据集节点类别数、测试集
"""
def prepare_data(dataset):
    # 加载数据集
    g, n_classes = load_graphgallery_data(dataset)
    feature_shape = g.ndata['features'].shape[1]
    train_g, val_g, test_g = split_graph_different_ratio(g, frac_list=[0.3, 0.2, 0.5], ratio=query_ratio)
    return feature_shape, n_classes, test_g


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
代理模型初始化
输入：数据集名称、特征形状、数据集类别；代理模型名称，代理模型隐藏层节点数；目标模型名称，目标模型隐藏层节点数
输出：代理模型，代理模型模型参数； 分类器，detached分类器。
"""
def surrogate_model_init(dataset, feature_shape, n_classes, surrogate_model_name, surrogate_model_dim, target_model_name, target_model_dim):
    # 加载模型参数，如果目标模型不是GIN，则model_args是没有用到的
    model_args = pickle.load(open('./surrogate_' + surrogate_model_name + "_" + str(surrogate_model_dim) +
                                  '_target_' + target_model_name +"_"+str(target_model_dim)+ '/model_args', 'rb'))
    model_args = model_args.__dict__
    model_args["gpu"] = gpu
    # 如果代理模型是GIN，则创建一个GIN模型并加载其状态字典；否则，直接从文件中加载目标模型
    if surrogate_model_name == 'gin':
        print('surrogate model: ', target_model_name)
        surrogate_model = GIN(feature_shape,
                              model_args['num_hidden'],
                              n_classes,
                              model_args['num_layers'],
                              F.relu,
                              model_args['batch_size'],
                              model_args['num_workers'],
                              model_args['dropout'])
                            
        surrogate_model.load_state_dict(torch.load('./surrogate_' + surrogate_model_name + "_" + str(surrogate_model_dim) +
                                  '_target_' + target_model_name +"_"+str(target_model_dim)+ "/surrogate_model_" + surrogate_model_name + "_" + dataset,
                                                map_location=torch.device('cpu')))
    else:
        surrogate_model = torch.load('./surrogate_' + surrogate_model_name + "_" + str(surrogate_model_dim) +
                                  '_target_' + target_model_name +"_"+str(target_model_dim)+ "/surrogate_model_" + surrogate_model_name + "_" + dataset,
                                                map_location=torch.device('cpu'))
    
    # 加载classifier和detached_classifier
    classifier = torch.load('./surrogate_' + surrogate_model_name + "_" + str(surrogate_model_dim) +
                                  '_target_' + target_model_name +"_"+str(target_model_dim)+ '/classifier')
    detached_classifier = torch.load('./surrogate_' + surrogate_model_name + "_" + str(surrogate_model_dim) +
                                  '_target_' + target_model_name +"_"+str(target_model_dim)+'/detached_classifier')
    return surrogate_model, model_args, classifier, detached_classifier


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
    return test_acc, pred, class_acc


"""
测试代理模型
输入：代理模型名称、代理模型、代理模型参数；分类器，detached分类器；测试集
输出：代理模型准确率、代理模型预测结果
"""
def evaluate_surrogate_model(surrogate_model_name,  surrogate_model, surrogate_model_args, classifier, detached_classifier, test_g):
    # 加载代理模型
    surrogate_model = surrogate_model.to(device)
    surrogate_model_args["gpu"] = gpu
    if surrogate_model_name == "gat":
        acc_surrogate, preds_surrogate, embds_surrogate,class_acc = evaluate_gat_surrogate(surrogate_model,
                                                                             classifier,
                                                                             test_g,
                                                                             test_g.ndata['features'],
                                                                             test_g.ndata['labels'],
                                                                             test_g.nodes(),
                                                                             surrogate_model_args["batch_size"],
                                                                             surrogate_model_args["head"],
                                                                             device)
    elif surrogate_model_name == "gin":
        acc_surrogate, preds_surrogate, embds_surrogate,class_acc = evaluate_gin_surrogate(surrogate_model,
                                                                             classifier,
                                                                             test_g, test_g.ndata['features'],
                                                                             test_g.ndata['labels'],
                                                                             test_g.nodes(),
                                                                             surrogate_model_args["batch_size"], device)
    elif surrogate_model_name == "sage":
        acc_surrogate, preds_surrogate, embds_surrogate,class_acc = evaluate_sage_surrogate(surrogate_model,
                                                                              classifier,
                                                                              test_g,
                                                                              test_g.ndata['features'],
                                                                              test_g.ndata['labels'],
                                                                              test_g.nodes(),
                                                                              surrogate_model_args["batch_size"],
                                                                              device)
    else:
        print("wrong recovery-from value")
        sys.exit()
    _acc = detached_classifier.score(embds_surrogate.clone().detach().cpu().numpy(),
                                 test_g.ndata['labels'])
    _predicts = detached_classifier.predict_proba(
        embds_surrogate.clone().detach().cpu().numpy())
    return _acc, _predicts, class_acc


"""
计算代理模型对于目标模型的保真率
输入：代理模型预测结果_predicts，目标模型预测结果pred
输出：保真率
"""
def evaluate_fidelity(_predicts, pred):
    _fidelity = compute_fidelity(torch.from_numpy(
    _predicts).to(device), pred.to(device))
    return _fidelity

"""
计算代理模型对于目标模型按类别的保真率
如目标模型认为有200个节点为2类，代理模型认为这200个节点中2类有180个，则保真率为90%
输入：代理模型预测结果_predicts，目标模型预测结果pred，数据集类别数n_classes
输出：每类别的保真率字典
"""
def evaluate_fidelity_by_class(_predicts, pred, n_classes):
    # 将_predicts转换为torch张量
    _predicts = torch.from_numpy(np.array(_predicts))

    class_fidelity = {}
    class_target_pred_num = [0] * n_classes
    class_surrogate_pred_right_num = [0] * n_classes
    target_pred_label = torch.argmax(pred, dim=1).tolist()
    surrogate_pred_label = torch.argmax(_predicts, dim=1).tolist()
    for i in range(len(target_pred_label)):
        class_target_pred_num[target_pred_label[i]] += 1
        if target_pred_label[i] == surrogate_pred_label[i]:
            class_surrogate_pred_right_num[target_pred_label[i]] += 1
    for i in range(n_classes):
        class_fidelity[i] = class_surrogate_pred_right_num[i]/class_target_pred_num[i]
    return class_fidelity


# 参数设置
gpu = 1
if gpu >= 0:
    device = th.device('cuda:%d' % gpu)
else:
    device = th.device('cpu')
query_ratio = 1.0

def execute():
    target_model_name = "gat"
    surrogate_model_name = "sage"
    dataset = "citeseer_full"
    target_model_dim = 256
    surrogate_model_dim = 256

    # 加载数据集
    feature_shape, n_classes, test_g = prepare_data(dataset)

    # 初始化 目标模型
    target_model, target_model_args = target_model_init(dataset, feature_shape, n_classes, target_model_name, target_model_dim)
    # 计算目标模型准确率
    test_acc, pred, target_class_acc = evaluate_target_model(target_model_name, target_model, target_model_args, test_g)
    del target_model
    del target_model_args

    # 初始化 代理模型
    surrogate_model, surrogate_model_args, classifier, detached_classifier = surrogate_model_init(dataset, feature_shape, n_classes, surrogate_model_name, surrogate_model_dim, target_model_name, target_model_dim)
    # 计算代理模型准确率
    _acc, _predicts,surrogate_class_acc = evaluate_surrogate_model(surrogate_model_name,  surrogate_model, surrogate_model_args, classifier, detached_classifier, test_g)
    del surrogate_model
    del surrogate_model_args
    del classifier
    del detached_classifier

    # 计算二者的保真率
    _fidelity = evaluate_fidelity(_predicts, pred)
     
    # 计算每类别的保真率
    class_fidelity = evaluate_fidelity_by_class(_predicts, pred, n_classes)

    print(class_fidelity)
    print(target_class_acc)
    print(surrogate_class_acc)
    print(test_acc.item())
    print(_acc)
    print(_fidelity)
    

execute()