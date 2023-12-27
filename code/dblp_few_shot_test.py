import argparse
import torch as th
from src.utils import load_graphgallery_data, split_graph
from graphgallery.datasets import NPZDataset
from src.gat import run_gat_target
from src.gin import run_gin_target
from src.sage import run_sage_target, evaluate_sage_target
import pickle
import os
import json
import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing
import random
import scipy.io as sio
import dgl

argparser = argparse.ArgumentParser("multi-gpu training")
argparser.add_argument('--gpu', type=int, default=1,
                       help="GPU device ID. Use -1 for CPU training")
argparser.add_argument('--target-model', type=str, default='gat')
argparser.add_argument('--dataset', type=str, default='dblp',
                       help="['dblp', 'pubmed', 'citeseer_full', 'coauthor_phy', 'acm' 'amazon_photo']")
argparser.add_argument('--num-epochs', type=int, default=100)
argparser.add_argument('--num-hidden', type=int, default=256)
argparser.add_argument('--num-layers', type=int, default=3)
argparser.add_argument('--fan-out', type=str, default='10,10,10')
argparser.add_argument('--batch-size', type=int, default=256)
argparser.add_argument('--val-batch-size', type=int, default=256)
argparser.add_argument('--log-every', type=int, default=20)
argparser.add_argument('--eval-every', type=int, default=100)
argparser.add_argument('--lr', type=float, default=0.001)
argparser.add_argument('--dropout', type=float, default=0.5)
argparser.add_argument('--num-workers', type=int, default=8,
                       help="Number of sampling processes. Use 0 for no extra process.")
argparser.add_argument('--inductive', action='store_true',
                       help="Inductive learning setting")
argparser.add_argument('--graphgallery', action='store_true', default=False)
argparser.add_argument('--save-pred', type=str, default='')
argparser.add_argument('--head', type=int, default=4)
argparser.add_argument('--wd', type=float, default=0)
args, _ = argparser.parse_known_args()
# params for sage
if args.target_model == "sage":
    args.inductive = True
    args.fan_out = '10,25'
# datasets = ['dblp', 'pubmed', 'citeseer', 'coauthor_phy', 'acm' 'amazon_photo']
# print(datasets)


if args.gpu >= 0:
    device = th.device('cuda:%d' % args.gpu)
else:
    device = th.device('cpu')


valid_num_dic = {'Amazon_clothing': 17, 'Amazon_eletronics': 36, 'dblp': 27}


def load_and_transfer_dblp_dataset():
    dataset_source = "dblp"
    n1s = []
    n2s = []
    for line in open("/home/data/ycx/my_program/GNN_Stealing/code/datasets/upload/dblp/{}_network".format(dataset_source)):
        n1, n2 = line.strip().split('\t')
        n1s.append(int(n1))
        n2s.append(int(n2))

    num_nodes = max(max(n1s),max(n2s)) + 1
    adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                                 shape=(num_nodes, num_nodes))


    data_train = sio.loadmat("/home/data/ycx/my_program/GNN_Stealing/code/datasets/upload/dblp/{}_train.mat".format(dataset_source))
    train_class = list(set(data_train["Label"].reshape((1,len(data_train["Label"])))[0]))
    

    data_test = sio.loadmat("/home/data/ycx/my_program/GNN_Stealing/code/datasets/upload/dblp/{}_test.mat".format(dataset_source))
    class_list_test = list(set(data_test["Label"].reshape((1,len(data_test["Label"])))[0]))


    labels = np.zeros((num_nodes,1))
    labels[data_train['Index']] = data_train["Label"]
    labels[data_test['Index']] = data_test["Label"]

    features = np.zeros((num_nodes,data_train["Attributes"].shape[1]))
    features[data_train['Index']] = data_train["Attributes"].toarray()
    features[data_test['Index']] = data_test["Attributes"].toarray()

    class_list = []
    for cla in labels:
        if cla[0] not in class_list:
            class_list.append(cla[0])  # unsorted

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(labels):
        id_by_class[cla[0]].append(id)

    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)

    degree = np.sum(adj, axis=1)
    degree = th.FloatTensor(degree)

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    features = th.FloatTensor(features)
    labels = th.LongTensor(np.where(labels)[1])

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    class_list_valid = random.sample(train_class, valid_num_dic[dataset_source])

    class_list_train = list(set(train_class).difference(set(class_list_valid)))

    return adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class 


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def convert_to_dgl_graph(adj, features, labels):
    adj = adj.coalesce()  # Ensure that the adjacency matrix is in COO format

    num_nodes = adj.shape[0]

    # Create a DGL graph
    dgl_graph = dgl.DGLGraph()
    dgl_graph.add_nodes(num_nodes)
    dgl_graph.add_edges(adj.indices()[0], adj.indices()[1])  # Use indices of the adjacency matrix

    # Set node features and labels
    dgl_graph.ndata['features'] = features
    dgl_graph.ndata['labels'] = labels

    # Add self loops to the graph
    dgl_graph = dgl.add_self_loop(dgl_graph)

    return dgl_graph

def convert_dblp_dataset(adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class):
    adj = adj.coalesce()  # Coalesce the adjacency matrix tensor
    adj = th.sparse_coo_tensor(adj.indices(), adj.values(), adj.shape)  # Convert to a sparse tensor
    features = th.FloatTensor(features)
    labels = th.LongTensor(labels)

    dgl_graph = convert_to_dgl_graph(adj, features, labels)
    num_classes = len(th.unique(labels))

    return dgl_graph, num_classes

# Load data and preprocessing
# g, n_classes = load_graphgallery_data(args.dataset)
adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class = load_and_transfer_dblp_dataset()
g, n_classes = convert_dblp_dataset(adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class)
in_feats = g.ndata['features'].shape[1]
labels = g.ndata['labels']

train_g, val_g, test_g = split_graph(g, frac_list=[0.7, 0.2, 0.1])

# 计算训练集类别分布
train_label_list = train_g.ndata['labels'].numpy().tolist()
num_class = {}
for i in range(len(train_label_list)):
    # num_class.setdefault(key, default) 获取指定key对应的value。如果key不存在,会设置key-value键值对,value为default指定的值
    num_class[train_label_list[i]] = num_class.setdefault(train_label_list[i], 0) + 1
sorted_dict = {}
sorted_keys = sorted(num_class.keys())

for key in sorted_keys:
   sorted_dict[key] = num_class[key]


print("训练集节点数   验证集节点数   测试集节点数")
print(train_g.number_of_nodes(), val_g.number_of_nodes(), test_g.number_of_nodes())

train_g.create_formats_()
val_g.create_formats_()
test_g.create_formats_()

# Train the target model
# print(args)
# exit()
SAVE_PATH = './target_model_%s_%s/' % (args.target_model, args.num_hidden)
SAVE_NAME = 'target_model_%s_%s' % (args.target_model, args.dataset)
os.makedirs(SAVE_PATH, exist_ok=True)

if args.target_model == "gat":
    data = train_g, val_g, test_g, in_feats, labels, n_classes, g, args.head
    target_model = run_gat_target(args, device, data)
    # th.save(target_model, SAVE_PATH + SAVE_NAME)

elif args.target_model == "gin":
    data = train_g, val_g, test_g, in_feats, labels, n_classes
    target_model = run_gin_target(args, device, data)
    # th.save(target_model.state_dict(), SAVE_PATH + SAVE_NAME)

elif args.target_model == "sage":
    # print("##########")
    data = in_feats, n_classes, train_g, val_g, test_g
    target_model = run_sage_target(args, device, data)
    # th.save(target_model, SAVE_PATH + SAVE_NAME)

else:
    raise ValueError("target-model should be gat, gin, or sage")


# Save model args
# pickle.dump(args, open(SAVE_PATH + 'model_args', 'wb'))


print(json.dumps(sorted_dict)) 
print("Finish")



