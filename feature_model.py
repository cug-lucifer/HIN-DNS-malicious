#目标 对决策树，SVM，DNN等方法进行测试
import numpy as np
import get_domain
import sklearn
import csv
from sklearn import svm
from sklearn import model_selection
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier,IsolationForest
from keras.models import Sequential
from keras.layers import Dense,Input
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

def visu_train_history(train_history,train_metrics,validation_metric):
	plt.plot(train_history.history[train_metrics])
	plt.plot(train_history.history[validation_metric])
	plt.title('Train_History')
	plt.ylabel('train_metrics')
	plt.xlabel('epoch')
	plt.legend(['train','validatio'],loc='upper left')
	plt.show()
def SVM(X_train, X_test, y_train, y_test):
    SVM_Model = svm.SVC(C=1.0)
    SVM_Model.fit(X_train, y_train)
    # 训练好的参数
    # print('Coefficients:%s \n\nIntercept %s' % (SVM_Model.coef_, SVM_Model.intercept_))
    predict_results = SVM_Model.predict(X_test)
    # 利用测试数据评判模型
    print(accuracy_score(predict_results, y_test))
    conf_mat = confusion_matrix(y_test, predict_results)
    print(conf_mat)
    print(classification_report(y_test, predict_results))


def DecisionTree(X_train, X_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    predict_results = clf.predict(X_test)
    print(accuracy_score(predict_results, y_test))
    conf_mat = confusion_matrix(y_test, predict_results)
    print(conf_mat)
    print(classification_report(y_test, predict_results))

def RandomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    predict_results = clf.predict(X_test)
    print(accuracy_score(predict_results, y_test))
    conf_mat = confusion_matrix(y_test, predict_results)
    print(conf_mat)
    print(classification_report(y_test, predict_results))

def IForest(X_train, X_test, y_test):
    ilf = IsolationForest(n_estimators=2,max_samples=8,contamination=0.1,n_jobs=-1)
    ilf.fit(X_train)
    predict_results = ilf.predict(X_test)
    print(predict_results)
    print(accuracy_score(predict_results, y_test))
    conf_mat = confusion_matrix(y_test, predict_results)
    print(conf_mat)
    print(classification_report(y_test, predict_results))

def DNN(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Input(shape=[11]))
    model.add(Dense(units=11,kernel_initializer ='uniform',
				  bias_initializer ='zero',
				  activation = 'relu'))
    model.add(Dense(units=8, activation='relu'))
    # 指定输出层,输出层的units就等于需要分类的种类
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    train_history = model.fit(x=X_train,
                              y=y_train,
                              validation_split=0.01,  # 将训练集数据分成两部分，20%用作验证集
                              epochs=50,
                              batch_size=10,
                              verbose=1  # 显示参数 0/1/2 0不显示；1显示进度条；2表示显示最多细节
                              )
    # 显示准确率
    #visu_train_history(train_history, 'acc', 'val_acc')
    # 显示损失函数
    #visu_train_history(train_history, 'loss', 'val_loss')

    y_pre_test = model.predict(X_test)
    label_test = y_test
    tot_num=len(y_pre_test)
    for threshold in np.arange(0, 1, 0.1):
        true_num = 0
        TP = 0
        FP = 0
        for i in range(tot_num):
            if (y_pre_test[i] >= threshold and label_test[i] == 1.0):
                true_num += 1
                TP += 1
            if (y_pre_test[i] < threshold and label_test[i] == 0.0):
                true_num += 1
                FP += 1
        print('threshold =', threshold, true_num, tot_num, true_num * 1.0 / tot_num)
        print(TP, FP)
        print(sum(label_test))
if __name__ == '__main__':

    i=1
    dict={}
    X_train=[]
    Y_train=[]
    X_test=[]
    Y_test = []
    X_Benion_train=[]
    X_Benion_test=[]
    X=[]
    Y=[]
    dataconPath='D:/毕设/DataSet/datacon_dns/datacon_dns/DNS_3/answer/label.csv'
    with open(dataconPath,encoding='utf-8') as f:
        for row in csv.reader(f):
            dns=row[0]
            if row[2] == 'white':
                Y.append(0.0)
            else:
                Y.append(1.0)

            '''dns=''.join(dns.split('['))
            dns = ''.join(dns.split(']'))
            print(dns)'''
            feature = get_domain.get_features(dns)
            X.append(feature)

    '''with open(trainfilepath,encoding='utf-8') as f:
        for row in csv.reader(f):
            #print(i)
            i+=1
            #print(row[0],row[1])
            feature = get_domain.get_features(row[1])
            dict[i]={'label':row[0],'feature':feature}
            if row[0]=='0':
                X_Benion_train.append(feature)
            X_train.append(feature)
            Y_train.append(int(row[0]))
    with open(testfilepath,encoding='utf-8') as f:
        for row in csv.reader(f):
            #print(i)
            i+=1
            #print(row[0],row[1])
            feature = get_domain.get_features(row[1])
            dict[i]={'label':row[0],'feature':feature}
            X_test.append(feature)
            Y_test.append(int(row[0]))
            if row[0]=='0':
                X_Benion_test.append(feature)
    '''
    #X = X_train+X_test
    #Y = Y_train + Y_test
    X_train, X_test, Y_train, Y_test= model_selection.train_test_split(X,Y,test_size=0.2)


    X_train = np.array(X_train,dtype=np.float)
    X_test = np.array(X_test, dtype=np.float)
    Y_train = np.array(Y_train, dtype=np.float)
    Y_test = np.array(Y_test, dtype=np.float)
    '''print('SVM')
    SVM(X_train, X_test, Y_train, Y_test)
    print('决策树')
    DecisionTree(X_train, X_test, Y_train, Y_test)
    print('随机森林')
    RandomForest(X_train, X_test, Y_train, Y_test)'''
    '''print('孤立森林')
    IForest(X_Benion_train, X_Benion_test,
            np.ones([len(X_Benion_test)],dtype=np.float))'''
    print('DNN')
    DNN(X_train, X_test, Y_train, Y_test)