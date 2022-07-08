import get_domain
import csv
import matplotlib.pyplot as plt
import numpy as np

def CCDF(data):
    Data = np.array(data)
    cdf = Data.cumsum(0)
    l = cdf[-1]
    cdf = 1.0*cdf / l
    ccdf = 1-cdf
    return ccdf

def dict2CCDF(d):
    k_vs=[]
    for k,v in d.items():
        k_vs.append([k,v])
    k_vs.sort()
    ks = []
    vs = []
    for k_v in k_vs:
        ks.append(k_v[0])
        vs.append(k_v[1])
    ccdf = CCDF(vs)
    print(len(ks),len(ccdf))
    return ks,ccdf

def drawccdflabels(X,Y):
    legends = []
    plt.plot(X[0],Y[0], color='blue', linestyle='-')
    plt.plot(X[1],Y[1], color='red', linestyle=':')

    legends.append('合法FQDN')
    legends.append('恶意FQDN')

    plt.xlabel('X: FQDN中平均标签长度')
    plt.xlim(0, 64)
    plt.xticks(np.arange(0, 64, 2))
    plt.grid(True)
    plt.ylabel('CCDF: P(最大标签长度>X)')
    plt.ylim((0, 1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend(legends)
    plt.show()


def analysisData(path,mode=0):
    feature_dict={'LenOfTot':{},'LenOfUp':{},'LenOfDig':{},'LenOfSup':{},'Entrophy':{},'NumOfLabel':{},
                  'MaxLenOfLabel':{},'AvgLenOfLabel':{}}
    def adddict(d,a):
        if d.get(a)!=None:
            d[a]+= 1
        else:
            d[a] = 1

    feature_dict['NumOfLabel'][0] = 0
    feature_dict['MaxLenOfLabel'][0] = 0
    feature_dict['AvgLenOfLabel'][0] = 0
    with open(path, encoding = 'utf-8') as f:
        for row in csv.reader(f):
            domain = row[mode]
            #print(domain)
            try:
                labels = get_domain.Labels(domain)
            except IndexError:
                print(domain)
                continue
            adddict(feature_dict['NumOfLabel'],labels[0])
            adddict(feature_dict['MaxLenOfLabel'], labels[1])
            adddict(feature_dict['AvgLenOfLabel'], labels[2])
    print(feature_dict)
    return feature_dict

benion_path='F:\数据集\Cosico-top-1m-benion/top-1m-2022-3-4.csv'
malcious_path = 'F:\数据集\Maclous/ExfiltrationAttackFQDNs.csv'

benion_dic=analysisData(benion_path,1)
malcious_dic = analysisData(malcious_path,0)
#label_num_X_benion, label_num_Y_benion= dict2CCDF(benion_dic['NumOfLabel'])
#label_num_X_malcious, label_num_Y_malcious= dict2CCDF(malcious_dic['NumOfLabel'])
#MaxLenOfLabel_X_benion, MaxLenOfLabel_Y_benion= dict2CCDF(benion_dic['MaxLenOfLabel'])
#MaxLenOfLabel_X_malcious, MaxLenOfLabel_Y_malcious= dict2CCDF(malcious_dic['MaxLenOfLabel'])
AvgLenOfLabel_X_benion, AvgLenOfLabel_Y_benion= dict2CCDF(benion_dic['AvgLenOfLabel'])
AvgLenOfLabel_X_malcious, AvgLenOfLabel_Y_malcious= dict2CCDF(malcious_dic['AvgLenOfLabel'])

drawccdflabels([AvgLenOfLabel_X_benion,AvgLenOfLabel_X_malcious],[AvgLenOfLabel_Y_benion,AvgLenOfLabel_Y_malcious])

