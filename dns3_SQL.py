import csv
import numpy as np
from matplotlib import pyplot as plt
import pymssql

def GetIPNum(path):
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
    return ip

def getFQDNandLABEL(path):
    labelPath = path + '/label.csv'
    label={}
    fqdn_white=[]
    fqdn_black=[]
    with open(labelPath, encoding='utf-8') as f:
        for row in csv.reader(f):
            if row[0] == 'encoded_fqdn':
                continue
            if row[2]!='white':
                fqdn_black.append(row[1])
                label[row[1]]=1
            else:
                fqdn_white.append(row[1])
                label[row[1]]=0
    return label,fqdn_black,fqdn_white

def getXY_SQL(FQDNs,IPs,cursor):
    X=[]
    Y=[]
    for i in range(len(FQDNs)):
        print('fqdn',i)
        fqdn=FQDNs[i]
        cursor.execute('select distinct encoded_ip from access where fqdn_no = %s',(fqdn))
        rows = cursor.fetchall()
        for row in rows:
            X.append(int(row[1][5:]))
            Y.append(IPs[row[0]])
    return X,Y
def getXY_Rows(rows,ip):
    X=[]
    Y=[]
    for row in rows:
        #if row[2]==1 or row[2] == 28:
            X.append(int(row[1][5:]))
            Y.append(ip[row[0]])
    return X,Y


def drawscatter(X,Y):
    plt.scatter(X,Y,s=1,linewidths=0.1,alpha=0.6,c='blue')
    #plt.scatter(X2, Y2, s=1, linewidths=0.1, alpha=0.6, c='red')
    #legends = ['合法FQDN']
    plt.xlabel('FQDN编号')
    plt.xlim(0,20000)
    plt.xticks(np.arange(0,20000,2000))

    plt.ylabel('IP编号')
    plt.ylim(0, 100000)
    plt.yticks(np.arange(0,100000,10000))

    #plt.legend(legends)
    plt.show()

if __name__ == '__main__':
    path1 = 'D:/毕设/DataSet/datacon_dns/datacon_dns/DNS_3/question'
    path2 = 'D:/毕设/DataSet/datacon_dns/datacon_dns/DNS_3/answer'
    label,fqdn_white,fqdn_black=getFQDNandLABEL(path2)
    print('label,fqdn readed')
    ipNum = GetIPNum(path1)
    print(len(ipNum))
    print('ip readed')
    conn = pymssql.connect(host='localhost', server='背道而驰', database='datacon_DNS', charset='cp936')
    cursor = conn.cursor()
    fqdn_=fqdn_black
    for i in range(len(fqdn_)):
        fqdn_[i]='\''+fqdn_[i]+'\''
    s = ','.join(fqdn_)
    #print(s)
    cursor.execute('select distinct encoded_ip, fqdn_no from access where date = \'20200531\' and fqdn_no in ('+s+') ')
    rows=cursor.fetchall()
    #print(rows)
    X,Y = getXY_Rows(rows,ipNum)
    drawscatter(X,Y)