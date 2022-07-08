import pyshark
import numpy as np
import csv


def exTLD(domain):
    labels = domain.split('.')
    l = len(labels)
    newdomain = ''.join(labels[:l - 2])
    return newdomain


def CntOfCh(Subdomain):
    totCnt = 0
    UpCnt = 0
    NumCnt = 0
    SupCnt = 0  # 特殊字符
    for ch in Subdomain:
        totCnt += 1
        if ch.isdigit():
            NumCnt += 1
        elif ch.isupper():
            UpCnt += 1
        elif ch.islower():
            continue
        SupCnt += 1
    return totCnt, UpCnt, NumCnt, SupCnt


def calc_ent(x):
    """
        calculate shanno ent of x
    """
    x_value_list = set([x[i] for i in range(len(x))])
    ent = 0.0
    for x_value in x_value_list:
        p = float(len(x[x == x_value])) / 64
        logp = np.log2(p)
        ent -= p * logp
    return ent


def Labels(domain):
    labels = domain.split('.')
    NumofLabels = len(labels)
    MaxLabelLength = 0
    TotLen = 0
    for label in labels:
        l = len(label)
        MaxLabelLength = max(l, MaxLabelLength)
        TotLen += l
    AvergeLabelLength = TotLen / NumofLabels
    return NumofLabels, MaxLabelLength, AvergeLabelLength

def LMW(dns):
    cnt=0
    Max=0
    num=0
    tot=0
    for i in range(len(dns)):
        if dns[i] == '[':
            cnt=i
        if dns[i] == ']':
            Max=max(Max,i-cnt)
            num+=1
            tot+=i-cnt
    if tot == 0:
        avg=0
    else:
        avg=tot/num
    return Max,num,avg,tot

def get_features(domain):
    dns = domain
    domain = ''.join(domain.split('['))
    domain = ''.join(domain.split(']'))
    LongestMeaningfulWord,NumMeaningWord,AvgMeaningWord,MWRadio=LMW(dns)
    SubDomain = exTLD(domain)
    # Count of Character
    totCntOfCharacters = len(domain)
    SubCntofCh, CntOfUpCh, CntOfNum, CntofSup = CntOfCh(domain)
    # Entropy
    Entropy = calc_ent(domain)
    # Labels
    NumofLabels, MaxLabelLength, AvergeLabelLength = Labels(domain)

    # print(totCntOfCharacters,CntOfUpCh,CntOfNum)
    # print(Entropy)
    # print(NumofLabels,MaxLabelLength,AvergeLabelLength)

    return [totCntOfCharacters,SubCntofCh,CntOfUpCh,CntOfNum,CntofSup,NumofLabels,MaxLabelLength,AvergeLabelLength,
            LongestMeaningfulWord,NumMeaningWord,AvgMeaningWord]
    #return [totCntOfCharacters, SubCntofCh, CntOfUpCh, CntOfNum, Entropy, NumofLabels, MaxLabelLength,AvergeLabelLength]


def check(domain, target='DNStest.com'):
    if target in domain:
        return True
    else:
        return False

def get_flink_features(rows):
    flint_dict = {}
    #fqdn_ip_dict ={}
    fqdn_no_set=set([])
    for row in rows:
        fqdn_no = row[0]
        flintType = row[1]
        encoded_value = row[2]
        requestCnt = row[3]
        fqdn_no_set.add(fqdn_no)
        #if fqdn_ip_dict.get(fqdn_no) == None:
        #    fqdn_ip_dict[fqdn_no] = [set([]),set([])]
        if flint_dict.get(fqdn_no) == None:
            flint_dict[fqdn_no] = {}
        if flint_dict[fqdn_no].get(flintType) == None:
            flint_dict[fqdn_no][flintType] = requestCnt
        #if flintType == 1:
        #    fqdn_ip_dict[fqdn_no][0].add(encoded_value)
        #elif flintType == 28:
        #    fqdn_ip_dict[fqdn_no][1].add(encoded_value)
    flint_feature = {}
    for fqdn in fqdn_no_set:
        NumofflintType=0
        totcnt=0
        ipv4_6cnt=0
        for flinttype,reqcnt in flint_dict[fqdn].items():
            NumofflintType+=1
            totcnt+=reqcnt
            if flinttype == 1 or flinttype == 28:
                ipv4_6cnt += reqcnt
        ipv4_6cnt_redio=ipv4_6cnt*1.0/totcnt
        #numOfipv4add = len(fqdn_ip_dict[fqdn][0])
        #numOfipv6add = len(fqdn_ip_dict[fqdn][1])
        #flint_feature[fqdn]=[NumofflintType,ipv4_6cnt_redio,numOfipv4add,numOfipv6add]
        flint_feature[fqdn] = [NumofflintType, ipv4_6cnt_redio]
    return flint_feature

if __name__ == '__main__':
    import pymssql
    conn = pymssql.connect(host='localhost', server='背道而驰', database='datacon_DNS', charset='cp936')
    cursor = conn.cursor()
    cursor.execute('select fqdn_no, flintType, encoded_value, requestCnt from flint')
    rows=cursor.fetchall()
    flint_feature=get_flink_features(rows)
    print(flint_feature)