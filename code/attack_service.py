import time
from dgl import load_graphs
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
import json
import argparse
import torch
import numpy as np
from core.model_handler import ModelHandler
from celery import Celery
from flask import Flask, url_for, jsonify, request
from flask import Flask
from flask_cors import CORS
import requests

attack_service = Flask(__name__)


# 解决有关进程的异常
torch.multiprocessing.set_start_method('spawn')
# 解决跨域
cors = CORS(attack_service, resources=r'/*')
# celery配置
# 配置消息代理的路径，如果是在远程服务器上，则配置远程服务器中redis的URL
attack_service.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
# 要存储 Celery 任务的状态或运行结果时就必须要配置
attack_service.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
# 初始化Celery
celery = Celery('attack_service', broker=attack_service.config['CELERY_BROKER_URL'])
# 将Flask中的配置直接传递给Celery
celery.conf.update(attack_service.config)


torch.set_num_threads(1)
torch.manual_seed(0)

DATASETS = ['citeseer_full']
RESPONSES = ['projection', 'prediction', 'embedding']

argparser = argparse.ArgumentParser("multi-gpu training")
argparser.add_argument('--gpu', type=int, default=-1,
                       help="GPU device ID. Use -1 for CPU training")
argparser.add_argument('--dataset', type=str, default='citeseer_full')
argparser.add_argument('--num-epochs', type=int, default=200)
argparser.add_argument('--transform', type=str, default='TSNE')
argparser.add_argument('--num-layers', type=int, default=2)
argparser.add_argument('--fan-out', type=str, default='10,50')
argparser.add_argument('--batch-size', type=int, default=1000)
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


surrogate_projection_accuracy = []
surrogate_prediction_accuracy = []
surrogate_embedding_accuracy = []

surrogate_projection_fidelity = []
surrogate_prediction_fidelity = []
surrogate_embedding_fidelity = []

target_accuracy = []


def idgl(config):
    model = ModelHandler(config)
    model.train()
    test_metrics, adj = model.test()
    return adj


def load_dataset():
    dataset_name = str(args.dataset)
    if dataset_name == 'elliptic':
        graph_list, _ = load_graphs('datasets/elliptic_from_5_to_10.bin')
        g = graph_list[0]
        n_classes = 2
    else:
        g, n_classes = load_graphgallery_data(args.dataset)
    return g, n_classes


def load_model_args():
    model_args = pickle.load(open('./target_model_' + args.target_model +
                                  '_' + str(args.target_model_dim) + '/model_args', 'rb'))
    model_args = model_args.__dict__
    model_args["gpu"] = args.gpu
    return model_args


def apply_model_args(g, n_classes, model_args):
    if args.target_model == 'gin':
        print('target model: ', args.target_model)
        target_model = GIN(g.ndata['features'].shape[1],
                           model_args['num_hidden'],
                           n_classes,
                           model_args['num_layers'],
                           F.relu,
                           model_args['batch_size'],
                           model_args['num_workers'],
                           model_args['dropout'])
        target_model.load_state_dict(torch.load('./target_model_' + args.target_model + '_' + str(
            args.target_model_dim) + '/' + './target_model_gin_' + args.dataset,
                                                map_location=torch.device('cpu')))
    else:
        target_model = torch.load('./target_model_' + args.target_model + '_' + str(
            args.target_model_dim) + '/' + './target_model_' + args.target_model + '_' + args.dataset,
                                  map_location=torch.device('cpu'))
    return target_model


def get_query_graph(train_g):
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


def preprocess_query_response(train_g, val_g, test_g, query_preds, query_embs, G_QUERY):
    # preprocess query response
    if args.recovery_from == 'prediction':
        print(args.dataset, args.recovery_from,
              'round {}'.format(str(args.round_index)))
        data = train_g.ndata['features'].shape[1], query_preds.shape[1], train_g, val_g, test_g, query_preds
    elif args.recovery_from == 'embedding':
        print(args.dataset, args.recovery_from,
              'round {}'.format(str(args.round_index)))
        data = train_g.ndata['features'].shape[1], query_preds.shape[1], train_g, val_g, test_g, query_embs
    elif args.recovery_from == 'projection':
        print(args.dataset, args.recovery_from,
              'round {}'.format(str(args.round_index)))
        tsne_embs = projection(query_embs.clone().detach().cpu().numpy(
        ), G_QUERY.ndata['labels'], transform_name=args.transform, gnn=args.target_model, dataset=args.dataset)
        tsne_embs = torch.from_numpy(tsne_embs.values).float().to(device)
        data = train_g.ndata['features'].shape[1], query_preds.shape[1], train_g, val_g, test_g, tsne_embs
    else:
        print("wrong recovery-from value")
        sys.exit()
    return data


def build_and_evaluate_surrogate_model(data, surrogate_model_filename, test_g):
    """
    构建代理模型 -> 训练代理模型 -> 评估代理模型
    """
    # which surrogate model to build
    if args.surrogate_model == 'gin':
        print('surrogate model: ', args.surrogate_model)
        model_s, classifier, detached_classifier = run_gin_surrogate(
            args, device, data, surrogate_model_filename)
        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_gin_surrogate(model_s,
                                                                                 classifier,
                                                                                 test_g, test_g.ndata['features'],
                                                                                 test_g.ndata['labels'],
                                                                                 test_g.nodes(),
                                                                                 args.batch_size, device)


    elif args.surrogate_model == 'gat':
        print('surrogate model: ', args.surrogate_model)
        model_s, classifier, detached_classifier = run_gat_surrogate(
            args, device, data, surrogate_model_filename)
        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_gat_surrogate(model_s,
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
        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_sage_surrogate(model_s,
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

    # 计算模型在测试集上的精度
    _acc = detached_classifier.score(embds_surrogate.clone().detach().cpu().numpy(),
                                     test_g.ndata['labels'])
    # 计算模型在测试集上的预测结果
    _predicts = detached_classifier.predict_proba(
        embds_surrogate.clone().detach().cpu().numpy())

    return _acc, _predicts


def save_result(_fidelity, _acc, test_acc):
    # which output to save
    if args.recovery_from == 'prediction':
        surrogate_prediction_fidelity.append(_fidelity)
        surrogate_prediction_accuracy.append(_acc)
    elif args.recovery_from == 'embedding':
        surrogate_embedding_fidelity.append(_fidelity)
        surrogate_embedding_accuracy.append(_acc)
    elif args.recovery_from == 'projection':
        surrogate_projection_fidelity.append(_fidelity)
        surrogate_projection_accuracy.append(_acc)
    else:
        print("wrong recovery-from value")
        sys.exit()
    OUTPUT_FOLDER = './results_acc_fidelity/results_%s_%d_%s_%d' % (
        args.target_model, args.target_model_dim, args.surrogate_model, args.num_hidden)
    if not os.path.isdir(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    if args.structure == 'original':
        _filename = OUTPUT_FOLDER + '/' + args.dataset + '_original.txt'
    elif args.structure == 'idgl':
        _filename = OUTPUT_FOLDER + '/' + args.dataset + '_idgl.txt'
    else:
        print("wrong structure params... stop!")
        sys.exit()

    with open(_filename, 'a') as wf:
        wf.write("%s,%d,%s,%d,%s,%d,%f,%f,%f,%f\n" % (args.target_model,
                                                      args.target_model_dim,
                                                      args.surrogate_model,
                                                      args.num_hidden,
                                                      args.recovery_from,
                                                      args.round_index,
                                                      args.query_ratio,
                                                      test_acc,
                                                      _acc,
                                                      _fidelity))


def query_target_model(target_model, G_QUERY, model_args):
    # query target model with G_QUERY
    # 这段代码根据目标模型类型,调用相应的evaluate函数,来查询该目标模型在G_QUERY上的预测结果、嵌入结果和准确率
    if args.target_model == 'sage':
        query_acc, query_preds, query_embs = evaluate_sage_target(target_model,
                                                                  G_QUERY,
                                                                  G_QUERY.ndata['features'],
                                                                  G_QUERY.ndata['labels'],
                                                                  G_QUERY.nodes(),
                                                                  args.batch_size,
                                                                  device)
    elif args.target_model == 'gin':
        query_acc, query_preds, query_embs = evaluate_gin_target(target_model,
                                                                 G_QUERY,
                                                                 G_QUERY.ndata['features'],
                                                                 G_QUERY.ndata['labels'],
                                                                 G_QUERY.nodes(),
                                                                 args.batch_size,
                                                                 device)
    elif args.target_model == 'gat':
        query_acc, query_preds, query_embs = evaluate_gat_target(target_model,
                                                                 G_QUERY,
                                                                 G_QUERY.ndata['features'],
                                                                 G_QUERY.ndata['labels'],
                                                                 G_QUERY.nodes(),
                                                                 model_args['val_batch_size'],
                                                                 model_args['head'],
                                                                 device)
    return query_acc, query_embs, query_preds


def evaluate_target_model(target_model, test_g, model_args):
    if args.target_model == 'sage':
        test_acc, pred, embs = evaluate_sage_target(target_model,
                                                    test_g,
                                                    test_g.ndata['features'],
                                                    test_g.ndata['labels'],
                                                    test_g.nodes(),
                                                    args.batch_size,
                                                    device)
    elif args.target_model == 'gat':
        test_acc, pred, embs = evaluate_gat_target(target_model,
                                                   test_g,
                                                   test_g.ndata['features'],
                                                   test_g.ndata['labels'],
                                                   test_g.nodes(),
                                                   model_args['val_batch_size'],
                                                   model_args['head'],
                                                   device)

    elif args.target_model == 'gin':
        test_acc, pred, embs = evaluate_gin_target(target_model,
                                                   test_g,
                                                   test_g.ndata['features'],
                                                   test_g.ndata['labels'],
                                                   test_g.nodes(),
                                                   args.batch_size,
                                                   device)
    return test_acc, pred, embs


# 远程访问版
def query_target_model_service(G_QUERY):
    t = G_QUERY.ndata['_ID'].tolist()
    emb_list = []
    pred_list = []
    right_number = 0
    for i in range(len(t)):
        node_idx = t[i]
        while True:  # 添加一个无限循环
            url = f'http://10.176.22.10:6020/query_node?node_idx={node_idx}'
            response = requests.get(url)
            if response.status_code == 200:
                response_data = response.json()
                print("已成功请求到第"+str(i)+"个节点" + str(node_idx) + "的预测，共计有" + str(len(t)) + "个节点")
                # 从json中拿到数组，转换为张量，然后拼接到一块
                node_emb = torch.Tensor(response_data['emb'])
                node_pred = torch.Tensor(response_data['pred'])
                pred_tag = response_data['pred_tag']
                real_tag = response_data['real_tag']
                if pred_tag == real_tag:
                    right_number = right_number + 1
                emb_list.append(node_emb)
                pred_list.append(node_pred)
                break  # 如果请求成功，退出无限循环
            else:
                print('请求失败，状态码：', response.status_code)
                print('正在尝试重新请求...')
    
    # 将列表转换为一个二维的tensor
    emb_tensor = torch.stack(emb_list)
    pred_tensor = torch.stack(pred_list)
    
    return right_number/len(t), emb_tensor, pred_tensor    


# 远程访问版
def evaluate_target_model_service(test_g):
    t = test_g.ndata['_ID'].tolist()
    emb_list = []
    pred_list = []
    right_number = 0
    for i in range(len(t)):
        node_idx = t[i]
        while True:  # 添加一个无限循环
            url = f'http://10.176.22.10:6020/query_node?node_idx={node_idx}'
            response = requests.get(url)
            if response.status_code == 200:
                response_data = response.json()
                print("已成功请求到第"+str(i)+"个节点" + str(node_idx) + "的预测，共计有" + str(len(t)) + "个节点")
                # 从json中拿到数组，转换为张量，然后拼接到一块
                node_emb = torch.Tensor(response_data['emb'])
                node_pred = torch.Tensor(response_data['pred'])
                pred_tag = response_data['pred_tag']
                real_tag = response_data['real_tag']
                if pred_tag == real_tag:
                    right_number = right_number + 1              
                emb_list.append(node_emb)
                pred_list.append(node_pred)
                break  # 如果请求成功，退出无限循环
            else:
                print('请求失败，状态码：', response.status_code)
                print('正在尝试重新请求...')
    
    # 将列表转换为一个二维的tensor
    emb_tensor = torch.stack(emb_list)
    pred_tensor = torch.stack(pred_list)
    
    return right_number/len(t), emb_tensor, pred_tensor  


@celery.task(bind=True, name='attack_service')
def attack_task(self, params):
    # 过滤为空的参数
    params = {k: v for k, v in params.items() if v is not None}
    # 统一修改args
    for k, v in params.items():
      setattr(args, k, v)   
    
    g, n_classes = load_dataset() # 划分数据集
    # args.query_ratio: 1.0 means we use 30% target dataset as the G_QUERY. 0.5 means 15%...
    train_g, val_g, test_g = split_graph_different_ratio(g, frac_list=[0.3, 0.2, 0.5], ratio=args.query_ratio)   
    G_QUERY = get_query_graph(train_g) # 构造查询图（train_g, train_g去边, IDGL重构）
    
    print(G_QUERY.ndata['_ID'])

    if args.structure != 'original': # 构造训练图
        print("using idgl reconstructed graph") # 如果图是离散的，那么在训练代理模型之前，先对其进行修复
        train_g = G_QUERY
    
    # 为图创建不同的稀疏矩阵格式
    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()
    
    # 查询目标模型，获取准确率、嵌入结果和预测结果
    # 嵌入结果包含样本的结构信息,但没有类别标签；预测结果包含样本的类别标签,但不一定包含结构信息
    query_acc, query_embs, query_preds = query_target_model_service(G_QUERY)
    query_embs = query_embs.to(device)
    query_preds = query_preds.to(device)
    
    data = preprocess_query_response(train_g, val_g, test_g, query_preds, query_embs, G_QUERY) # 对查询响应的数据进行预处理

    
    # 设置代理模型存储位置
    surrogate_model_filename = './surrogate_model'
    # 训练代理模型，并返回在测试集上的准确率
    _acc, _predicts = build_and_evaluate_surrogate_model(data, surrogate_model_filename, test_g)
    


    # 获得目标模型在测试集上的准确率和预测
    test_acc, embs, preds  = evaluate_target_model_service(test_g)
    
    target_accuracy.append(test_acc)
    # 计算保真度
    _fidelity = compute_fidelity(torch.from_numpy(
        _predicts).to(device), preds.to(device))
    
    print("目标模型标签准确率：" + str(query_acc))
    print("代理模型测试集准确率: " + str(_acc))
    print("目标模型测试集准确率: " + str(test_acc))
    print("保真度：" + str(_fidelity))
    
    # save_result(_fidelity, _acc, test_acc)



@attack_service.route('/execute', methods=['GET'])
def start_func():
    params = request.get_json()
    task = attack_task.apply_async(kwargs={'params': params}) # 异步调用
    return jsonify({'message': '长任务启动成功.', 'task_id': task.id}), 202


@attack_service.route('/tasks', methods=['GET'])
def list_tasks():
    i = celery.control.inspect() # 获取所有任务
    all_tasks = {
        'active_tasks': i.active(),  # 获取当前正在执行的任务
        'queued_tasks': i.reserved(),  # 获取当前排队的任务
        'scheduled_tasks': i.scheduled()  # 获取当前scheduled的任务
    }
    return jsonify(all_tasks), 200

"""
@attack_service.route('/query', methods=['GET'])
def task_status():
    task_id = request.args.get('task_id')
    task = attack_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': '正在等待...'
        }
    elif task.state == "GMS001" or "GMS002" or "GMS003" or "GMS004" or "GMS005" or "GMS006" or "GMS007" :
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            # todo
            # 'status': task.info.get('status', ''),
        }
    else:
        # 后端执行任务出现了一些问题
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # 报错的具体异常
        }
    return jsonify(response)
"""



if __name__ == '__main__':
    attack_service.run(host="0.0.0.0", port=6016)
