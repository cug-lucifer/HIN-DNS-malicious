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


if __name__ == '__main__':
    file_path = 'F:\数据集\自建恶意数据集\iodine/iodine.pcap'
    cap = pyshark.FileCapture(input_file=file_path, display_filter='dns')
    csvfile = open("csv_test.csv", "w", newline="", encoding='utf-8')
    writer = csv.writer(csvfile)
    for pkt in cap:
        test = str(pkt.dns.qry_name)
        if check(test):
            print(test)
            writer.writerow([test])
    csvfile.close()
    cap.close()
