import csv
import numpy as np
from matplotlib import pyplot as plt
import pymssql
import torch as th
import torch.nn as nn
import dgl
import get_feature


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
            fqdn_feature.append(get_feature.get_features(row[0]))

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

bad_clients=set([])
bad_domains=set([])
one_domain=set([])

def find_bad_client_and_DNS_node(rows):
    K_d=0.25
    K_c=0.50
    client_query={}
    DNS_queried={}
    domain=set([])
    client=set([])
    for row in rows:
        d=row[0]
        c=row[1]
        domain.add(row[0])
        client.add(row[1])
        if client_query.get(c)!=None:
            client_query[c].add(d)
        else:
            client_query[c]=set([])
            client_query[c].add(d)

        if DNS_queried.get(d)!=None:
            DNS_queried[d].add(c)
        else:
            DNS_queried[d]=set([])
            DNS_queried[d].add(c)


    client_num=len(client)
    domain_num=len(domain)

    sorted(client_query.items(),key=lambda item:item[1])
    i=0
    n=int(client_num*0.001)
    for k,v in client_query.items():
        if i<n or len(v) < 3:
            bad_clients.add(k)
            i+=1
    for k,v in DNS_queried.items():
        if len(v)*1.0/client_num >= K_d:
            bad_domains.add(k)
        if len(v)<=1:
            one_domain.add(k)


if __name__ == '__main__':
    path1 = 'D:/毕设/DataSet/datacon_dns/datacon_dns/DNS_3/question'
    path2 = 'D:/毕设/DataSet/datacon_dns/datacon_dns/DNS_3/answer'
    ip_Dic,ip_TotNum = Get_IP_Num_from_CSV(path1)
    fqdn_feature = Get_FQDN_Feature_from_CSV(path1)
    DNS_num=len(fqdn_feature)
    label_Dic = Get_LABEL_from_CSV(path2)
    conn = pymssql.connect(host='localhost', server='背道而驰', database='datacon_DNS', charset='cp936')
    cursor = conn.cursor()
    cursor.execute('select distinct date, hour from access')
    date_hours= cursor.fetchall()
    '''for date_hour in date_hours:
        cursor.execute('select distinct fqdn_no, encoded_ip from access where date = {} and hour = {}'.format(date_hour[0],date_hour[1]))
        rows = cursor.fetchall()
        find_bad_client_and_DNS_node(rows)
        print(date_hour)'''
    cursor.execute('select distinct fqdn_no, encoded_ip from access')
    rows = cursor.fetchall()
    find_bad_client_and_DNS_node(rows)
    print(len(bad_clients),len(bad_domains))
    print(bad_clients)
    print(bad_domains)
    print(len(one_domain),one_domain)
    bad_clients=list(bad_clients)
    with open('./ConDNSDataset/save/bad_clients_0001.txt','w') as f:
        for bad_client in bad_clients:
            f.write(bad_client)

