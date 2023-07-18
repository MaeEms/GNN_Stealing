import dgl
import torch as th
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import scipy as sp
from scipy.sparse import csr_matrix
import numpy as np
from dgl.data.utils import save_graphs


# 加载 elliptic: 读取 csv 格式的文件
def load_elliptic_from_csv():
    # 交易的类别，有'unknown', '1', '2' 三种
    classes = pd.read_csv("../datasets/elliptic/elliptic_txs_classes.csv", index_col='txId')
    # 交易之间的有向边
    edgelist = pd.read_csv("../datasets/elliptic/elliptic_txs_edgelist.csv", index_col='txId1')
    # 交易的特征
    features = pd.read_csv("../datasets/elliptic/elliptic_txs_features.csv", header=None, index_col=0)
    return classes, edgelist, features


# 截取指定时间范围内的 features
def collect_features_by_ts(features, start_ts, end_ts):
    # 用于收集指定时间步长范围内的所有features
    features_chosen_by_ts = []
    for ts in range(start_ts, end_ts):
        # 从 features 中 选出 features[1]即表示时间步长的feature 等于 ts+1 的
        # 因为features中的时间步长从1-50，这里是从0-49
        features_ts = features[features[1] == ts + 1]
        features_chosen_by_ts.append(features_ts)
    # 将收集的 features 合并成一张大 DataFrame （但没有总表大）
    features_ts = pd.concat(features_chosen_by_ts)
    return features_ts


# 从 classes 中截取带标签的部分， 并和和带标签的节点 节点id 组成的列表 一同返回
def get_labelled_classes(classes):
    # 在 classes 中只保留标记过的节点 （交易号，标签） 标签：1 or 2
    labelled_classes = classes[classes['class'] != 'unknown']
    #  从上一步获取的 labelled_classes 中提取交易号
    labelled_tx = list(labelled_classes.index)
    return labelled_classes, labelled_tx


# 获取一个列表，记录了 ts时间范围内 所有labelled的节点id列表
def get_labelled_tx_ts(features_ts, labelled_tx):
    # 获取此 时间步长范围内 所有的 交易节点id 列表
    tx_ts = list(features_ts.index)
    # 从上一步 交易节点id 列表 中选取被标记过的 （在 labelled_tx 中的）
    labelled_tx_ts = [tx for tx in tx_ts if tx in set(labelled_tx)]
    return labelled_tx_ts


# 获取一个矩阵，这个矩阵中所有节点都是 labelled，且为 ts范围 内的节点
def get_labelled_adj_mat_ts(labelled_tx_ts, edgelist):
    length = len(labelled_tx_ts)
    # 所有交易的临接矩阵
    # 我们只会填入这个时间步长范围内的交易，这些交易是有标签可用于训练的
    # 先创建一个 length * length 的临接矩阵（该时间步长范围内，所有带标记节点都装得下）
    adj_mat = pd.DataFrame(np.zeros((length, length)), index=labelled_tx_ts, columns=labelled_tx_ts)
    # 这行代码的作用是将与标记节点相关的边筛选出来，并将其存储在一个新的DataFrame对象中
    # 通过将 "edgelist" DataFrame 的索引与 "labelled_tx_ts" Series 的唯一值交集进行比较，确定与标记节点相关的边
    edgelist_labelled_ts = edgelist.loc[edgelist.index.intersection(labelled_tx_ts).unique()]

    # 对于其中的每一条边，在矩阵 adjmat 中填入1，得到临接矩阵
    for i in range(edgelist_labelled_ts.shape[0]):
        adj_mat.loc[edgelist_labelled_ts.index[i], edgelist_labelled_ts.iloc[i]['txId2']] = 1
    # 此时adj_mat是一个大的稀疏矩阵
    # 获取一个小的稀疏矩阵 矩阵中的横纵表示的是ts步长范围内 所有被标记的节点ID，值代表它们是否有边
    adj_mat_ts = adj_mat.loc[labelled_tx_ts, labelled_tx_ts]
    return adj_mat_ts


# 获取以上矩阵 tx节点的 feature, 存在字典里, 键为节点ID,值为节点剩余的features
def get_features_labelled_ts(features, labelled_tx_ts):
    # 构建 features_dict
    # 获取当前ts范围内，所有被标记节点的feature
    features_l_ts = features.loc[labelled_tx_ts]
    index_list = list(features_l_ts.index)
    features_l_ts_array = features_l_ts.to_numpy()
    features_dict = {}
    i = 0
    for row in features_l_ts_array:
        # todo th.tensor(row[1:], dtype=th.float32) 转 dataframe
        features_dict[int(index_list[i])] = row[1:]
        i = i + 1
    return features_dict


# 构建 labelled_dict, 将labelled_classes数据框转换为一个字典, 其中键是节点id, 值是节点类别
def get_labelled_classes_ts(labelled_classes, labelled_tx_ts):
    # 构建 labelled_dict
    # 将labelled_classes数据框转换为一个字典，其中键是节点id，值是节点类别
    diction = dict(zip(labelled_classes.index, labelled_classes['class']))
    labelled_dict = {}
    # 遍历labelled_tx_ts列表，并对于其中的每个节点id，将其类别添加到一个名为labelled_dict的字典中
    # 其中键是labelled_tx_ts中的节点id，值是相应的节点类别
    for node_id in labelled_tx_ts:
        labelled_dict[node_id] = np.long(diction[node_id])
    return labelled_dict


def load_elliptic_data(start_ts, end_ts):
    # Step 1
    # 加载 elliptic: 读取 csv 格式的文件
    classes, edgelist, features = load_elliptic_from_csv()

    # Step 2
    # 从 classes 中剔除“unknown”部分， 并返回有标签节点的列表 labelled_tx
    labelled_classes, labelled_tx = get_labelled_classes(classes)
    # 截取指定时间范围内 tx节点 的 features
    features_ts = collect_features_by_ts(features, start_ts, end_ts)

    # Step 3
    # 截取指定时间范围内 有标记的tx节点 的 id 列表
    labelled_tx_ts = get_labelled_tx_ts(features_ts, labelled_tx)

    # Step 4
    # 构建 labelled_dict, 获取所选时间步长范围内、所有被标记节点 节点ID 类别  组成的字典
    labelled_dict = get_labelled_classes_ts(labelled_classes, labelled_tx_ts)

    # 获取 指定 时间范围内 所有有标记的节点 构成的矩阵
    adj_mat_ts = get_labelled_adj_mat_ts(labelled_tx_ts, edgelist)
    # 获取以上矩阵 tx节点的 feature, 存在字典里, 键为节点ID,值为节点剩余的features
    features_dict = get_features_labelled_ts(features, labelled_tx_ts)

    # Step 5
    # 开始凑格式
    query_list = list(adj_mat_ts.index)
    # 这个sparse_matrix就是我要的
    sparse_matrix = csr_matrix(adj_mat_ts.values)

    nx_g = nx.from_scipy_sparse_matrix(sparse_matrix)

    for node_id, node_data in nx_g.nodes(data=True):
        node_data["features"] = features_dict[query_list[node_id]].astype(np.float32)
        # 分为两类 0 或 1
        node_data["labels"] = np.int_(labelled_dict[query_list[node_id]] - 1)

    dgl_graph = dgl.from_networkx(nx_g, node_attrs=['features', 'labels'])
    dgl_graph = dgl.add_self_loop(dgl_graph)
    save_graphs('../datasets/elliptic_from_5_to_10', dgl_graph)
    # return dgl_graph, 2

# 使用 load_elliptic_data(start_ts, end_ts) 来制作数据集
load_elliptic_data(5, 10)
