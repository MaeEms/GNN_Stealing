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
import requests
from core.model_handler import ModelHandler
from celery import Celery
from flask import Flask, url_for, jsonify, request
from flask import Flask
from flask_cors import CORS



torch.set_num_threads(1)
torch.manual_seed(0)


DATASETS = ['citeseer_full']
RESPONSES = ['projection', 'prediction', 'embedding']


argparser = argparse.ArgumentParser("multi-gpu training")
argparser.add_argument('--gpu', type=int, default=1,
                       help="GPU device ID. Use -1 for CPU training")
argparser.add_argument('--dataset', type=str, default='citeseer_full')
argparser.add_argument('--num-epochs', type=int, default=200)
argparser.add_argument('--transform', type=str, default='TSNE')
argparser.add_argument('--num-layers', type=int, default=2)
argparser.add_argument('--fan-out', type=str, default='10,50')
argparser.add_argument('--batch-size', type=int, default=400) # 之前是1000
argparser.add_argument('--log-every', type=int, default=20)
argparser.add_argument('--eval-every', type=int, default=50)
argparser.add_argument('--lr', type=float, default=0.001)
argparser.add_argument('--dropout', type=float, default=0.5)
argparser.add_argument('--num-workers', type=int, default=4,
                       help="Number of sampling processes. Use 0 for no extra process.")
argparser.add_argument('--inductive', action='store_true',
                       help="Inductive learning setting")
argparser.add_argument('--head', type=int, default=4)
argparser.add_argument('--wd', type=float, default=0)
argparser.add_argument('--target-model', type=str, default='sage')
argparser.add_argument('--target-model-dim', type=int, default=256)
argparser.add_argument('--surrogate-model', type=str, default='sage')
argparser.add_argument('--num-hidden', type=int, default=256)
argparser.add_argument('--recovery-from', type=str, default='embedding')
argparser.add_argument('--round_index', type=int, default=1)
argparser.add_argument('--query_ratio', type=float, default=1.0,
                       help="1.0 means we use 30% target dataset as the G_QUERY. 0.5 means 15%...")
argparser.add_argument('--structure', type=str, default='original')
argparser.add_argument('--delete_edges', type=str, default='no')

args, _ = argparser.parse_known_args()

args.inductive = True

if args.gpu >= 0:
    device = th.device('cuda:%d' % args.gpu)
else:
    device = th.device('cpu')

"""
    idgl重构图结构
"""
def idgl(config):
    model = ModelHandler(config)
    model.train()
    test_metrics, adj = model.test()
    return adj


"""
    将 train_g 处理为 G_QUERY
    输入:   train_g
    处理:   根据图结构是否完整的设置,对train_g进行边的删除和恢复来模拟
    输出:   G_QUERY
"""
def prepare_G_QUERY(train_g):
    if args.structure == 'original':
        G_QUERY = train_g
        # only use node to query
        if args.delete_edges == "yes":
            G_QUERY = delete_dgl_graph_edge(train_g)

    elif args.structure == 'idgl':
        config['dgl_graph'] = train_g
        config['cuda_id'] = args.gpu
        adj = idgl(config)
        adj = adj.clone().detach().cpu().numpy()
        if args.dataset in ['acm', 'amazon_cs']:
            adj = (adj > 0.9).astype(np.int)
        elif args.dataset in ['coauthor_phy']:
            adj = (adj >= 0.999).astype(np.int)
        else:
            adj = (adj > 0.999).astype(np.int)
        sparse_adj = sparse_csr_mat = sparse.csr_matrix(adj)
        G_QUERY = dgl.from_scipy(sparse_adj)
        G_QUERY.ndata['features'] = train_g.ndata['features']
        G_QUERY.ndata['labels'] = train_g.ndata['labels']
        G_QUERY = dgl.add_self_loop(G_QUERY)    
    else:
        print("wrong structure param... stop!")
        sys.exit()

    return G_QUERY


"""
    访问目标模型,得到预测
    输入:   目标模型访问api, 查询图
    处理:   访问url, 发送查询图, 接收目标模型对查询图的预测
    输出:   目标模型对查询图的预测
"""
def get_query_graph_pred(url,g_query):
    # 将test_g序列化为字节流
    serialized_g_query = pickle.dumps(g_query)
    # 发送POST请求到目标模型的HTTP接口
    url = 'http://10.176.22.10:6020/get_query_graph_pred'
    headers = {'Content-Type': 'application/octet-stream'}
    response = requests.post(url, data=serialized_g_query, headers=headers)
    g_query_pred_array = response.json()['query_preds']
    return torch.tensor(g_query_pred_array)


"""
    根据目标模型给出的标签，训练代理模型，并在测试集上评估代理模型
    输入:   data(包含训练集和目标模型给其打上的标签), 代理模型结果存储位置, 测试集
    处理:   先在训练集上执行训练, 再在测试集上执行测试
    输出:   代理模型在测试集上的准确率、预测preds、每类的准确率
"""
def train_and_evaluate_surrogate_model(data, surrogate_model_filename, test_g):
    # which surrogate model to build
    if args.surrogate_model == 'gin':
        print('surrogate model: ', args.surrogate_model)
        model_s, classifier, detached_classifier = run_gin_surrogate(
            args, device, data, surrogate_model_filename)
        acc_surrogate, preds_surrogate, embds_surrogate,class_acc = evaluate_gin_surrogate(model_s,
                                                                                classifier,
                                                                                test_g, test_g.ndata['features'],
                                                                                test_g.ndata['labels'],
                                                                                test_g.nodes(),
                                                                                args.batch_size, device)
    elif args.surrogate_model == 'gat':
        print('surrogate model: ', args.surrogate_model)
        model_s, classifier, detached_classifier = run_gat_surrogate(
            args, device, data, surrogate_model_filename)
        acc_surrogate, preds_surrogate, embds_surrogate,class_acc = evaluate_gat_surrogate(model_s,
                                                                                classifier,
                                                                                test_g,
                                                                                test_g.ndata['features'],
                                                                                test_g.ndata['labels'],
                                                                                test_g.nodes(),
                                                                                args.batch_size,
                                                                                args.head,
                                                                                device)


    elif args.surrogate_model == 'sage':
        print('surrogate model: ', args.surrogate_model)
        model_s, classifier, detached_classifier = run_sage_surrogate(
            args, device, data, surrogate_model_filename)
        acc_surrogate, preds_surrogate, embds_surrogate,class_acc = evaluate_sage_surrogate(model_s,
                                                                                classifier,
                                                                                test_g,
                                                                                test_g.ndata['features'],
                                                                                test_g.ndata['labels'],
                                                                                test_g.nodes(),
                                                                                args.batch_size,
                                                                                device)
    else:
        print("wrong recovery-from value")
        sys.exit()
    # 计算代理模型准确率
    _acc = detached_classifier.score(embds_surrogate.clone().detach().cpu().numpy(),
                                 test_g.ndata['labels'])
    _predicts = detached_classifier.predict_proba(
        embds_surrogate.clone().detach().cpu().numpy())
    
    return _acc, _predicts,class_acc

"""
    Compute the accuracy of prediction given the labels.
"""
def compute_target_acc(pred, labels):
    # tag
    labels = labels.long()
    class_acc = {}
    # 最多能够处理十分类问题
    class_label_num = [0]*10
    pred_label_num = [0]*10
    pred_labels = torch.argmax(pred, dim=1).tolist()
    for i in range(len(labels)):
        class_label_num[labels[i]] += 1
        if pred_labels[i] == labels[i]:
            pred_label_num[labels[i]] +=1
    for i in range(10):
        if class_label_num[i] != 0:
            class_acc[i] = pred_label_num[i]/class_label_num[i]
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred), class_acc


"""
    计算代理模型对于目标模型的保真率
    输入:   代理模型预测结果_predicts,目标模型预测结果pred
    输出:   保真率
"""
def evaluate_fidelity(_predicts, pred):
    _fidelity = compute_fidelity(torch.from_numpy(
    _predicts).to(device), pred.to(device))
    return _fidelity


"""
    计算代理模型对于目标模型按类别的保真率
    如目标模型认为有200个节点为2类, 代理模型认为这200个节点中2类有180个, 则保真率为90%
    输入:   代理模型预测结果_predicts, 目标模型预测结果pred, 数据集类别数n_classes
    输出:   每类别的保真率字典
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


def main_process():
    # 加载并划分数据集
    g, n_classes = load_graphgallery_data(args.dataset)
    train_g, val_g, test_g = split_graph_different_ratio(g, frac_list=[0.3, 0.2, 0.5], ratio=args.query_ratio)

    # 对 train_g 进行更新, 更新为攻击设置的场景（图结构完整，图结构不完整，图结构原本不完整但是idgl恢复）
    G_QUERY = prepare_G_QUERY(train_g)
    if args.structure != 'original':
        print("using idgl reconstructed graph")
        train_g = G_QUERY

    # 访问目标模型，获取目标模型对查询图的预测
    query_preds = get_query_graph_pred('http://10.176.22.10:6020/get_query_graph_pred',G_QUERY)
    torch.cuda.empty_cache()
    query_preds = query_preds.to(device)

    # 格式化
    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()

    surrogate_model_filename = './surrogate_model'

    # 默认目标模型返回的事preds的形式，构建data来进行训练
    data = train_g.ndata['features'].shape[1], query_preds.shape[1], train_g, val_g, test_g, query_preds

    # 获得了测试集上：代理模型的准确率、代理模型的预测、代理模型在各类节点上的准确率
    _acc, _predicts,surrogate_class_acc = train_and_evaluate_surrogate_model(data, surrogate_model_filename, test_g)
    _predicts = _predicts

    # 访问目标模型，获取目标模型对测试集的预测
    target_preds = get_query_graph_pred('http://10.176.22.10:6020/get_query_graph_pred',test_g)
    target_preds = target_preds.to(device)

    test_g_label = test_g.ndata['labels'][test_g.nodes()]

    # 计算目标模型在测试集上的准确率 和 按类别的准确率
    target_acc, target_class_acc = compute_target_acc(target_preds[test_g.nodes()],test_g_label.to(device))
    
    # 计算代理模型和目标模型之间的保真率
    _fidelity = evaluate_fidelity(_predicts, target_preds)
    # 计算每类别的保真率
    class_fidelity = evaluate_fidelity_by_class(_predicts, target_preds, n_classes)

    response = {
        'target_model_acc': target_acc.item(),
        'target_model_acc_by_class':target_class_acc,
        'surrogate_model_acc': _acc,
        'surrogate_model_acc_by_class':surrogate_class_acc,
        'fidelity': _fidelity,
        'fidelity_by_class':class_fidelity
    }

    print(response)



main_process()


