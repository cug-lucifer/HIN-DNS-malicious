import csv
import numpy as np
from matplotlib import pyplot as plt
import pymssql
import torch as th
import torch.nn as nn
import dgl
import get_feature

bad_client = []


with open('./ConDNSDataset/save/bad_clients_0001.txt','r') as f:
    bad_client=f.readlines()
def Get_IP_Num_from_CSV(path):
    IPv4Path = path + '/ip.csv'
    IPv6Path = path + '/ipv6.csv'
    ip={}
    i = 0
    with open(IPv4Path, encoding='utf-8') as f:
        for row in csv.reader(f):
            if row[0] == 'encoded_ip':
                continue
            ip[row[0]] = i
            i += 1
    with open(IPv6Path, encoding='utf-8') as f:
        for row in csv.reader(f):
            if row[0] == 'encoded_ip':
                continue
            ip[row[0]] = i
            i += 1
    return ip,i

def Get_FQDN_Feature_from_CSV(path):

    fqdnpath=path + '/fqdn.csv'
    fqdn_feature=[]
    i = 0
    with open(fqdnpath,encoding='utf-8') as f:
        for row in csv.reader(f):
            if row[0]=='encoded_fqdn':
                continue
            feature = get_feature.get_features(row[0])
            #flint_feature = flint_feature_dict.get(row[0])
            #if flint_feature == None:
            #    flint_feature = [0,0,0,0]

            fqdn_feature.append(feature)

    return fqdn_feature

def get_fqdn_num(fqdn_no):
    return int(fqdn_no[5:])

def Get_LABEL_from_CSV(path):
    labelPath = path + '/label.csv'
    label={}
    with open(labelPath, encoding='utf-8') as f:
        for row in csv.reader(f):
            if row[0] == 'encoded_fqdn':
                continue
            if row[2]!='white':
                label[row[1]]=1
            else:
                label[row[1]]=0
    return label

def GetAccessEdges(accesses):
    domain=[]
    client=[]
    client_dic={}
    i=0
    for access in accesses:
        if access[1] in bad_client:
            continue

        domain.append(get_fqdn_num(access[0]))
        c = client_dic.get(access[1])
        if c == None:
            c = i
            client_dic[access[1]] = i
            i += 1
        client.append(c)
    print(len(client_dic) , len(set(client)))
    return domain,client,client_dic

def Get_Domain_IP_Edges(rows):
    domain = []
    IP = []
    ipDic = {}
    i=0
    for row in rows:
        domain.append(get_fqdn_num(row[0]))
        c = ipDic.get(row[1])
        if c == None:
            c = i
            ipDic[row[1]]=i
            i += 1
        IP.append(c)
    return domain, IP, ipDic

def Get_Domain_CName_Edges(rows):
    domain=[]
    cname=[]
    i=0
    cdic={}
    for row in rows:
        domain.append(get_fqdn_num(row[0]))
        c = cdic.get(row[1])
        if c == None:
            c=i
            cdic[c]=i
            i+=1
        cname.append(c)
    return domain,cname,cdic

if __name__ == '__main__':
    path1 = 'D:/毕设/DataSet/datacon_dns/datacon_dns/DNS_3/question'
    path2 = 'D:/毕设/DataSet/datacon_dns/datacon_dns/DNS_3/answer'
    ip_Dic,ip_TotNum = Get_IP_Num_from_CSV(path1)

    conn = pymssql.connect(host='localhost', server='背道而驰', database='datacon_DNS', charset='cp936')
    cursor = conn.cursor()
    #cursor.execute('select fqdn_no, flintType, encoded_value, requestCnt from flint')
    #rows = cursor.fetchall()
    #flint_feature_dict = get_feature.get_flink_features(rows)
    fqdn_feature = Get_FQDN_Feature_from_CSV(path1)

    label_Dic = Get_LABEL_from_CSV(path2)

    cursor.execute('select distinct fqdn_no, encoded_ip from access')
    rows = cursor.fetchall()
    #print(rows[0])
    Domain_access, Client, clientDic = GetAccessEdges(rows)
    #print(ip_TotNum)
    cursor.execute('select distinct fqdn_no,encoded_value from flint where flintType = 1 or flintType = 28')
    rows = cursor.fetchall()
    Domain_flint, IP, ipDic = Get_Domain_IP_Edges(rows)
    #print(ip_TotNum)
    cursor.execute('select distinct fqdn_no,encoded_value from flint where flintType = 5')
    rows = cursor.fetchall()
    Domain_C, CName, CNameDic = Get_Domain_CName_Edges(rows)

    domain_nodes=len(fqdn_feature)
    graph=dgl.heterograph({
        ('domain', 'domain-client', 'client'):(th.tensor(Domain_access),th.tensor(Client)),
        ('client', 'client-domain', 'domain'): (th.tensor(Client), th.tensor(Domain_access)),
        ('domain', 'domain-ip', 'ip'):(th.tensor(Domain_flint),th.tensor(IP)),
        ('ip', 'ip-domain', 'domain'): (th.tensor(IP), th.tensor(Domain_flint)),
        ('domain', 'domain-cname', 'cname'):(th.tensor(Domain_C),th.tensor(CName)),
        ('cname', 'cname-domain', 'domain'):(th.tensor(CName),th.tensor(Domain_C))
    }, num_nodes_dict={'domain': domain_nodes,
                       'cname':len(CNameDic),
                       'ip':len(ipDic),
                       'client':len(clientDic)}
    )
    '''graph = dgl.heterograph({
        ('domain', 'domain-cname', 'cname'): (th.tensor(Domain_C), th.tensor(CName)),
        ('cname', 'cname-domain', 'domain'): (th.tensor(CName), th.tensor(Domain_C))
    }, num_nodes_dict={'domain': domain_nodes,'cname':CName_Num})'''

    graph.nodes['domain'].data['feature']=th.tensor(fqdn_feature[:domain_nodes],dtype=th.float)
    print(graph)
    print(graph.nodes['domain'])
    dgl.save_graphs('./model/graph/DNSGraph_badnodes_0001.dgl',graph)