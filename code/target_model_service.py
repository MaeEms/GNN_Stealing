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
from celery import Celery
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

DATASETS = ['citeseer_full']
RESPONSES = ['projection', 'prediction', 'embedding']
argparser = argparse.ArgumentParser("multi-gpu training")
argparser.add_argument('--gpu', type=int, default=-1,
                       help="GPU device ID. Use -1 for CPU training")
argparser.add_argument('--batch-size', type=int, default=1000)

args, _ = argparser.parse_known_args()
args.inductive = True
if args.gpu >= 0:
    device = th.device('cuda:%d' % args.gpu)
else:
    device = th.device('cpu')


def init_service(dataset, target_model_name, target_model_dim):
    # 加载数据集
    dataset_name = str(dataset)
    if dataset_name == 'elliptic':
        graph_list, _ = load_graphs('datasets/elliptic_from_5_to_10.bin')
        g = graph_list[0]
        n_classes = 2
    else:
        g, n_classes = load_graphgallery_data(dataset)
    # 加载模型参数，如果目标模型不是GIN，则model_args是没有用到的
    model_args = pickle.load(open('./target_model_' + target_model_name +
                                  '_' + str(target_model_dim) + '/model_args', 'rb'))
    model_args = model_args.__dict__
    model_args["gpu"] = args.gpu
    # 如果目标模型是GIN，则创建一个GIN模型并加载其状态字典；否则，直接从文件中加载目标模型
    if target_model_name == 'gin':
        print('target model: ', target_model_name)
        target_model = GIN(g.ndata['features'].shape[1],
                           model_args['num_hidden'],
                           n_classes,
                           model_args['num_layers'],
                           F.relu,
                           model_args['batch_size'],
                           model_args['num_workers'],
                           model_args['dropout'])
        target_model.load_state_dict(torch.load('./target_model_' + target_model_name + '_' + str(
            target_model_dim) + '/' + './target_model_gin_' + dataset,
                                                map_location=torch.device('cpu')))
    else:
        target_model = torch.load('./target_model_' + target_model_name + '_' + str(
            target_model_dim) + '/' + './target_model_' + target_model_name + '_' + dataset,
                                  map_location=torch.device('cpu'))
    return target_model, model_args, g, n_classes


def query_target_model(G_QUERY, target_model_name):
    label_probs = None
    # query target model with G_QUERY
    # 这段代码根据目标模型类型,调用相应的evaluate函数,来查询该目标模型在G_QUERY上的预测结果、嵌入结果和准确率
    if target_model_name == 'sage':
        query_acc, query_preds, query_embs, label_probs = evaluate_sage_target(target_model,
                                                                               G_QUERY,
                                                                               G_QUERY.ndata['features'],
                                                                               G_QUERY.ndata['labels'],
                                                                               G_QUERY.nodes(),
                                                                               args.batch_size,
                                                                               device)
    elif target_model_name == 'gin':
        query_acc, query_preds, query_embs = evaluate_gin_target(target_model,
                                                                 G_QUERY,
                                                                 G_QUERY.ndata['features'],
                                                                 G_QUERY.ndata['labels'],
                                                                 G_QUERY.nodes(),
                                                                 args.batch_size,
                                                                 device)
    elif target_model_name == 'gat':
        query_acc, query_preds, query_embs = evaluate_gat_target(target_model,
                                                                 G_QUERY,
                                                                 G_QUERY.ndata['features'],
                                                                 G_QUERY.ndata['labels'],
                                                                 G_QUERY.nodes(),
                                                                 model_args['val_batch_size'],
                                                                 model_args['head'],
                                                                 device)
    return query_embs, query_preds


def extract_node_subgraph(dataset_name, i, dgl_graph):
    # set `verbose=False` to avoid additional outputs
    data = NPZDataset(dataset_name, verbose=False)
    graph = data.graph

    # 第i行的邻接节点
    row_data = data.graph.adj_matrix[i, :].toarray()
    # 第i行值为1的节点索引
    row_indices = np.where(row_data > 0)[1]
    row_indices = np.insert(row_indices, 0, i)

    node_sub_graph = dgl_graph.subgraph(row_indices)

    if not 'features' in node_sub_graph.ndata:
        node_sub_graph.ndata['features'] = node_sub_graph.ndata['feat']
    if not 'labels' in node_sub_graph.ndata:
        node_sub_graph.ndata['labels'] = node_sub_graph.ndata['label']
        
    return node_sub_graph


def draw_sub_graph(node_idx, sub_graph):
    plt.clf()
    edges = sub_graph.edges()
    number = sub_graph.number_of_nodes()
    adj = np.zeros((number, number))  # 创建一个全0矩阵
    # 使用src和dst表示边
    for src, dst in zip(edges[0], edges[1]):
        adj[src, dst] = 1  # 设置边元件为1
    G = nx.Graph()
    for i in range(number):
        for j in range(number):
            if adj[i, j] > 0 and i != j:
                G.add_edge(i, j)
    nx.draw(G, node_size=300, node_color='red', with_labels=True)
    plt.savefig("images/"+"node_"+node_idx+"_adjacency.SVG")   # svg格式


# 设置目标模型参数
target_model, model_args, g, n_class = init_service(dataset="citeseer_full", target_model_name="gat",
                                                    target_model_dim=256)
target_model = target_model.to(device)


# 测试
# 共计4230 编号0——4229 [1390, 1586, 498, 1032, 565]的邻居节点最多 565 3456 2881 3444x 4021
@target_model_app.route('/query_node', methods=['GET'])
def query_func():
    node_idx = request.args.get('node_idx')
    sub_graph = extract_node_subgraph('citeseer_full', int(node_idx), g)

    # 绘图
    # draw_sub_graph(node_idx, sub_graph)

    query_embs, query_preds = query_target_model(sub_graph, 'gat')
    softmax_preds = F.softmax(query_preds[0], dim=0)

    # print("查询结点：" + str(node_idx))
    # print("节点嵌入:" + str(query_embs[0]))
    # print("预测结果:"+str(query_preds[0].tolist()))
    # print("概率分布：" + str(softmax_preds.tolist()))
    # print("预测标签：" + str(th.argmax(query_preds[0]).item()))
    print(str(node_idx) + "真实标签：" + str(sub_graph.ndata['labels'][0].item()))

    response = {
        'node_idx': node_idx,
        # 'emb': softmax_preds.tolist(),
        'emb': query_embs[0].tolist(),
        #'pred': th.argmax(query_preds[0]).item()
        'pred': query_preds[0].tolist(),
        'pred_tag': th.argmax(query_preds[0]).item(),
        'real_tag': sub_graph.ndata['labels'][0].item()
    }
    return jsonify(response), 200


if __name__ == '__main__':
    target_model_app.run(host="0.0.0.0", port=6020)
