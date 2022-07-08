import csv
import numpy as np
import torch as th
import torch.nn as nn
import dgl
from model import MyGNN,MyGNN_mean,MyGNN_GCN
import torch.nn.functional as F

label_dic={'botnet':0,'trojan':1,'apt':2,'gamble':3,'white':4}

def get_graph(path='./MyDNS.dgl'):
    [graph],_=dgl.load_graphs(path)
    print(type(graph))
    print(graph)
    return graph

def get_label(DNS_Nodes_Num,path='./ConDNSDataset/answer/label.csv'):

    DNS_Labels = th.zeros(DNS_Nodes_Num, dtype=th.float)
    DNS_Nodes_With_label_mask = th.zeros(DNS_Nodes_Num, dtype=th.bool)
    with open(path, encoding='utf-8') as f:
        i = -1
        for row in csv.reader(f):
            if i == -1:
                i += 1
                continue
            fqdn_no = row[1]
            label = row[2]
            DNS_num = int(fqdn_no[5:])
            DNS_Nodes_With_label_mask[DNS_num] = True
            if label != 'white':
                DNS_Labels[DNS_num] = 1.0
    return DNS_Labels,DNS_Nodes_With_label_mask

def train_GNN(GNN,epoches,graph,meta_path_dist,DNS_Nodes_With_label_mask,train_mask,DNS_labels,features):
    test_mask=~train_mask
    #GNN = MyHAN.GNN(meta_path_dist, len(features[0]), len(features[0]), 2, 0)
    opt = th.optim.Adam(GNN.parameters())
    # print(list(model.parameters()))
    import itertools
    loss_train=[]
    loss_list=[]
    for e in range(epoches):
        GNN.train()
        logits = GNN(graph, {'domain': features})
        logits = logits[DNS_Nodes_With_label_mask]

        #print(type(logits[train_mask]), len(logits[train_mask]),logits)
        loss = F.binary_cross_entropy(logits[train_mask], DNS_labels[train_mask])
        loss1= F.binary_cross_entropy(logits[test_mask], DNS_labels[test_mask])
        opt.zero_grad()
        loss.backward()
        print('epoch {0} : loss = {1} , test_loss = {2}'.format(e,loss.data,loss1.data))
        loss_list.append(loss.data.item())
        loss_train.append(loss.data)
        opt.step()

        if e % 50 == 0:
            y_pre_test = logits[test_mask]
            label_test =  DNS_labels[test_mask]
            tot_num = len(y_pre_test)
            true_num = 0
            #print(type(y_pre_test), type(label_test))

            for i in range(tot_num):
                if (y_pre_test[i] >= 0.5 and label_test[i] == 1.0) or (y_pre_test[i] < 0.5 and label_test[i] == 0.0):
                    true_num += 1

            print(true_num, tot_num, true_num * 1.0 / tot_num)
    print(loss_train)
    th.save(GNN, './model/save_model/DNSGNN_all_0_1000_GCN.pth')
    loss_list = np.array(loss_list)
    np.save('loss_list_0_1000_GCN.npy', loss_list)
    return GNN



if __name__ == '__main__':
    graph=get_graph('./model/graph/DNSGraph.dgl')
    DNS_Nodes_Num = graph.num_nodes('domain')
    DNS_Labels, DNS_Nodes_With_label_mask=get_label(DNS_Nodes_Num)
    DNS_labels_ = DNS_Labels[DNS_Nodes_With_label_mask]
    label_num = len(DNS_labels_)
    features = graph.nodes['domain'].data['feature']
    #print(len(features[0]))
    #train_mask = th.zeros(label_num, dtype=th.bool).bernoulli(0.8)

    meta_path_dist = {'d-c-d': ['domain-client', 'client-domain'],
                      'd-i-d': ['domain-ip', 'ip-domain'],
                      'd-o-d': ['domain-cname', 'cname-domain']
                      }
    #meta_path_dist = {'d-o-d': ['domain-cname', 'cname-domain']}
    #GNN_model = MyGNN.GNN(meta_path_dist, len(features[0]), len(features[0]), 2, 0)
    GNN_model = th.load('./model/save_model/DNSGNN_all_0_1000_dod.pth')
    #GNN_model = MyGNN_mean.GNN(meta_path_dist, len(features[0]), len(features[0]), 2, 0)
    #GNN_model = MyGNN_GCN.GNN(meta_path_dist, len(features[0]), len(features[0]), 2, 0)
    train_mask = th.load('./ConDNSDataset/save/train_mask__all_100_0001.dat')
    #train_GNN(GNN_model,1000, graph, meta_path_dist, DNS_Nodes_With_label_mask, train_mask, DNS_labels_, features)

    y_pre=GNN_model(graph,{'domain': features})
    y_pre=y_pre[DNS_Nodes_With_label_mask]


    y_pre_test=y_pre[~train_mask]
    label_test=DNS_labels_[~train_mask]
    tot_num=len(y_pre_test)

    threshold=0.5
    true_num = 0
    TP = 0
    TN = 0
    for i in range(tot_num):
        if (y_pre_test[i]>=threshold and label_test[i] == 1.0):
                true_num+=1
                TP += 1
        if  (y_pre_test[i]<threshold and label_test[i] == 0.0):
                true_num += 1
                TN += 1
    print('threshold =',threshold,true_num,tot_num,true_num*1.0/tot_num)
    FN=int(sum(label_test))-TP
    FP=tot_num-FN-TP-TN
    prec=1.0*TP/(FP+TP)
    recall = 1.0*TP / (FN + TP)
    F1=2.0*prec*recall/(prec+recall)
    print('Precision = {0}, Recall = {1}, F1 = {2}'.format(prec,recall,F1))
