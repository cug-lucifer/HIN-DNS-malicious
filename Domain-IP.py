import csv
import numpy as np
from matplotlib import pyplot as plt

def GetIPNum(path):
    IPv4Path = path + '/ip.csv'
    #IPv6Path = path + '/ipv6.csv'
    ip={}
    i = 0
    with open(IPv4Path, encoding='utf-8') as f:
        for row in csv.reader(f):
            if row[0] == 'encoded_ip':
                continue
            ip[row[0]] = i
            i += 1
    return ip

def NoOfFQDN2NumOfFQDN(NoOfFQDN):
    return int(NoOfFQDN[5:])

def GetFlintRel(path):
    FlintPath = path + '/flint.csv'
    d2i=[]
    with open(FlintPath, encoding='utf-8') as f:
        for row in csv.reader(f):
            if row[0] == 'fqdn_no':
                continue
            if row[1] == '1':
                d2i.append([row[0],row[2]])
    return d2i

def GetDomainLabel(path):
    Labelpath = path + '/label.csv'
    dld={}
    fqdndict={}
    i=0
    with open(Labelpath, encoding='utf-8') as f:
        for row in csv.reader(f):
            if row[0] == 'fqdn_no':
                continue
            dld[row[0]]=row[1]
            fqdndict[row[0]] = i
            i+=1
    print('Num of labeled FQDN: ',i)
    return dld,fqdndict

def getXY(Domain2IP,IPDict,DomainLabelDict,FQDNDict,NoOfFQDNDict):
    X_1=[]
    Y_1=[]
    X_0=[]
    Y_0=[]
    FQDN_1=[]
    FQDN_0=[]
    IP2FQDN0=[{}]*len(IPDict)
    IP2FQDN1=[{}]*len(IPDict)
    for d2i in Domain2IP:
        try:
            if DomainLabelDict[d2i[0]] == '1':
                #print(d2i)
                Y_1.append(IPDict[d2i[1]])
                X_1.append(NoOfFQDNDict[d2i[0]])
                #FQDN_1.append(FQDNDict[d2i[0]])
                #IP2FQDN1[IPDict[d2i[1]]].update({d2i[0]:1})
            else:
                #print(d2i)
                Y_0.append(IPDict[d2i[1]])
                X_0.append(NoOfFQDNDict[d2i[0]])
                #FQDN_0.append(FQDNDict[d2i[0]])
                #IP2FQDN0[IPDict[d2i[1]]].update({d2i[0]:1})
        except KeyError:
            #print(d2i[0],len(d2i[0]))
            continue
    return X_1,Y_1,X_0,Y_0,FQDN_1,FQDN_0,IP2FQDN1,IP2FQDN0

def drawscatter(X,Y,X2,Y2):
    plt.scatter(X,Y,s=1,linewidths=0.1,alpha=0.6,c='blue')
    plt.scatter(X2, Y2, s=1, linewidths=0.1, alpha=0.6, c='red')
    legends = ['合法FQDN']
    plt.xlabel('恶意FQDN')
    plt.xlim(0,2000)
    plt.xticks(np.arange(0,2000,500))

    plt.grid(True)
    plt.ylabel('IP编号')
    plt.ylim(0, 5000)
    plt.yticks(np.arange(0, 5000, 500))

    plt.legend(legends)
    plt.show()

def drawbar(X1):
    l = len(X1)
    print([len(x) for x in X1])
    plt.bar(range(l),[len(x) for x in X1],width=0.1,fc='red')
    plt.show()

def NoOfFQDN2FQDN(path):
    FQDNPath=path + '/fqdn.csv'
    FQDNDict = {}
    with open(FQDNPath, encoding='utf-8') as f:
        for row in csv.reader(f):
            if row[1] == 'fqdn_no':
                continue
            FQDNDict[row[1]] = row[0]
    return FQDNDict

def getFQDNAccess(path):
    AccessPath = path + ''

if __name__ == '__main__':
    FilesPath='D:/毕设/DataSet/datacon_dns/datacon_dns/DNS_2/question/train'
    IPDict=GetIPNum(FilesPath)
    Domain2IP=GetFlintRel(FilesPath)
    DomainLabelDict, NoOfFQDNDict = GetDomainLabel(FilesPath)
    FQDNDict = NoOfFQDN2FQDN(FilesPath)
    #print(Domain2IP)
    X_1, Y_1, X_0, Y_0, FQDN1, FQDN0, IP2FQDN1,IP2FQDN0= getXY(Domain2IP,IPDict,DomainLabelDict,FQDNDict,NoOfFQDNDict)

    Domain2IP.sort()
    #print(Domain2IP)
    #print(Y_0)
    drawscatter(X_1,Y_1,X_0,Y_0)
    #drawbar(IP2FQDN1)
    #drawbar(IP2FQDN0)
