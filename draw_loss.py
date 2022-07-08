import numpy as np
import matplotlib.pyplot as plt

loss_lists=['loss_list_0_1000_dcd.npy',
            'loss_list_0_1000_did.npy',
            'loss_list_0_1000_dod.npy',
            'loss_list_0_1000_mean.npy',
            'loss_list_0_1000_3p.npy']
l=[]
for loss_list in loss_lists:
    a=np.load('./'+loss_list,allow_pickle=True)
    l.append(a)
legends = []

x=range(1000)

plt.plot(x,l[0],c='blue')
plt.plot(x,l[1],c='red')
plt.plot(x,l[2],c='green')
plt.plot(x,l[3],c='orange')
plt.plot(x,l[4],c='black')
legends.append('D-C-D')
legends.append('D-I-D')
legends.append('D-R-D')
legends.append('均值方法')
legends.append('本文方法')
plt.xlabel('训练轮数')
plt.xlim(0,1000)
plt.xticks(np.arange(0,1001,100))
plt.ylabel('交叉熵损失')
plt.ylim(0,1)
plt.yticks(np.arange(0,1.1,0.1))
plt.legend(legends)
plt.grid(True)
plt.show()