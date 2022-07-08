import get_domain
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def CCDF(data):
    Data = np.array(data)
    cdf = Data.cumsum(0)
    l = cdf[-1]
    cdf = 1.0*cdf / l
    ccdf = 1-cdf
    return ccdf


def draw_CCDF_1(data):
    legends = []
    ccdf = CCDF(data)
    plt.plot(range(256),ccdf)
    legends.append('whatever')

    plt.title('CCDF total de registros' )
    plt.xlabel('#número de vendas')
    plt.xlim(0, 256)

    plt.grid(True)
    plt.ylabel('porcentagem (prob. de ocorrência)')
    plt.ylim((0,1))
    plt.yticks(np.arange(0, 1.0, 0.1), fontsize=24)
    plt.legend(legends)
    plt.show()

def draw_CCDF_2(data1,data2):
    legends = []

    Data1 = np.array(data1)
    Data2 = np.array(data2)
    cdf1=Data1.cumsum(0)
    cdf2 = Data2.cumsum(0)
    l = cdf1[-1]
    cdf1 = 1.0*cdf1 / l
    l = cdf2[-1]
    cdf2 = 1.0* cdf2 / l

    ccdf1 = 1-cdf1
    ccdf2 = 1 - cdf2

    plt.plot(range(256),ccdf1,color='blue',linestyle='-')
    plt.plot(range(256), ccdf2, color='red', linestyle=':')

    legends.append('合法DNS域名')
    legends.append('恶意DNS域名')

    plt.title('DNS域名字符数量分布图')
    plt.xlabel('X: 域名中字符数量')
    plt.xlim(0, 256)
    plt.xticks(np.arange(0, 256, 20))
    plt.grid(True)
    plt.ylabel('CCDF: P(字符数>X)')
    plt.ylim((0,1))
    plt.yticks(np.arange(0, 1.0, 0.1))
    plt.legend(legends)
    plt.show()

def draw_plot_1(x,y):#benion -  malcious :
    plt.plot(x,y,color='blue',linewidth=2,linestyle='-')
    plt.xlim((0, 256))
    plt.ylim((0,50000))
    # 设置x,y的坐标描述标签
    plt.xlabel("域名长度")
    plt.ylabel("数量")
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    plt.show()

def NumofCh(benion_path,malcious_path):
    lengthNum_benion = [0] * 256
    length_benion = []
    with open(benion_path, encoding='utf-8') as f:
        for row in csv.reader(f):
            l = len(row[1])
            if l == 0:
                continue
            lengthNum_benion[l] += 1
            length_benion.append(l)
    '''        if i > 10:
                break'''

    lengthNum_malcious = [0] * 256
    length_malcious = []
    with open(malcious_path, encoding='utf-8') as f:
        for row in csv.reader(f):
            l = len(row[0])
            if l == 0:
                continue
            lengthNum_malcious[l] += 1
            length_malcious.append(l)
    print(lengthNum_benion)
    print(lengthNum_malcious)
    draw_CCDF_2(lengthNum_benion, lengthNum_malcious)

def draw_CCDF_4(Tot,Up,Dig,Sup):
    legends = []

    ccdfTot = CCDF(Tot)

    ccdfUp = CCDF(Up)
    ccdfDig = CCDF(Dig)
    ccdfSup = CCDF(Sup)

    plt.plot(range(256),ccdfTot,color='blue',linestyle='-')
    plt.plot(range(256), ccdfUp, color='red', linestyle=':')
    plt.plot(range(256), ccdfDig, color='green', linestyle='-.')
    plt.plot(range(256), ccdfSup, color='yellow', linestyle='--')

    legends.append('字符数量')
    legends.append('大写字母数量')
    legends.append('数字数量')
    legends.append('特殊字符数量')

    #plt.title('合法DNS域名')
    plt.title('恶意DNS域名')
    plt.xlabel('X: 域名中字符数量')
    plt.xlim(0, 256)
    plt.xticks(np.arange(0, 256, 20))
    plt.grid(True)
    plt.ylabel('CCDF: P(字符数>X)')
    plt.ylim((0,1))
    plt.yticks(np.arange(0, 1.0, 0.1))
    plt.legend(legends)
    plt.show()

def featuresOfCh(path):
    benionNumOfTot = [0] * 256
    benionNumOfUp = [0] * 256
    benionNumOfDig = [0] * 256
    benionNumOfSup = [0] * 256
    with open(path, encoding='utf-8') as f:
        for row in csv.reader(f):
            #domain = row[1]
            domain = row[0]
            feature = get_domain.CntOfCh(domain)
            benionNumOfTot[feature[0]] += 1
            benionNumOfUp[feature[1]] += 1
            benionNumOfDig[feature[2]] += 1
            benionNumOfSup[feature[3]] += 1
    draw_CCDF_4(benionNumOfTot,benionNumOfUp,benionNumOfDig,benionNumOfSup)

def ccdfOfEnt(data):
    data=np.array(data)
    ents=[]
    nums=[]
    for d in data:
        ents.append(d[0])
        nums.append(d[1])
    ccdf = CCDF(nums)
    plt.plot(ents,ccdf)

    plt.xlabel('X: 域名的熵')
    plt.xlim(0, 6.5)
    plt.xticks(np.arange(0,6.5,0.5))

    plt.grid(True)
    plt.ylabel('CCDF: P(熵>X)')
    plt.ylim((0, 1.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.show()


def entrophyofDomain(path):
    EntOfDomain = {}
    with open(path, encoding='utf-8') as f:
        for row in csv.reader(f):
            domain=row[0]
            #print(domain)
            try:
                ent = get_domain.calc_ent(domain)
            except IndexError:
                #print(domain)
                continue
            if EntOfDomain.get(ent)!=None:
                EntOfDomain[ent]+=1
            else:
                EntOfDomain[ent]=1
    ents_num = []
    for ent,num in EntOfDomain.items():
        ents_num.append([ent,num])
    ents_num.sort()
    print(ents_num)
    ccdfOfEnt(ents_num)




benion_path='F:\数据集\Cosico-top-1m-benion/top-1m-2022-3-4.csv'
malcious_path = 'F:\数据集\Maclous/ExfiltrationAttackFQDNs.csv'



#featuresOfCh(benion_path)
#featuresOfCh(malcious_path)
#entrophyofDomain(benion_path)
#entrophyofDomain(malcious_path)

def ccdlentrophyofbe_mal(data1,data2):
    legends = []
    data1 = np.array(data1)
    ents1 = []
    nums1 = []
    for d in data1:
        ents1.append(d[0])
        nums1.append(d[1])
    ccdf = CCDF(nums1)
    data2 = np.array(data2)
    ents2 = []
    nums2 = []
    for d in data2:
        ents2.append(d[0])
        nums2.append(d[1])
    ccdf2 = CCDF(nums2)
    plt.plot(ents1, ccdf,color='blue',linestyle='-')
    plt.plot(ents2, ccdf2, color='red', linestyle=':')
    plt.xlabel('X: 域名的熵')
    plt.xlim(0, 6.5)
    plt.xticks(np.arange(0, 6.5, 0.5))

    plt.grid(True)
    plt.ylabel('CCDF: P(熵>X)')
    plt.ylim((0, 1.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    legends.append('合法DNS域名')
    legends.append('恶意DNS域名')
    plt.legend(legends)
    plt.show()
def entrophyofDomain2(path,type=0):
    EntOfDomain = {}
    with open(path, encoding='utf-8') as f:
        for row in csv.reader(f):
            if type == 0:
                domain = row[1]
            else:
                domain=row[0]
            #print(domain)
            try:
                ent = get_domain.calc_ent(domain)
            except IndexError:
                #print(domain)
                continue
            if EntOfDomain.get(ent)!=None:
                EntOfDomain[ent]+=1
            else:
                EntOfDomain[ent]=1
    ents_num = []
    for ent,num in EntOfDomain.items():
        ents_num.append([ent,num])
    ents_num.sort()
    print(ents_num)
    return ents_num

#ccdlentrophyofbe_mal(entrophyofDomain2(benion_path,0),entrophyofDomain2(malcious_path,1))
#print(get_domain.calc_ent('www.google.com'))
