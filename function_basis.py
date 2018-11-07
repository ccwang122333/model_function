# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 12:00:41 2018

@author: cc_privide
"""

#常用函数打包
#1.提取sql数据库数据
from sqlalchemy import create_engine
import pandas as pd
def sql_function(address):
    sql_engine=create_engine('postgresql://postgres:postgres@localhost:5432/mimic',echo=True)
    data=pd.read_sql_query(address,con=sql_engine,index_col=None, coerce_float=True, params=None, parse_dates=None,chunksize=None)
    return data
#i.e.:sql_funciton('select * from mimiciii.patients limit 100')

#2.归一化数据预处理
#from sklearn import preprocessing
#3.random data_orign
#from sklearn.utils import shuffle
#函数三个变量分别是原始数据，随机数种子（伪随机），n_sample输出样本数，默认none,输出行数
#shuffle(data,random_state,n_sample)
    
#4.统计data中个数据出现的次数
#from collections import Counter
#a=Counter(data)

#5.RandomForest对特征值进行评分
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
import numpy as np
def feature_value_RF(x_orign,y_orign,n_iter):
    label=pd.DataFrame(x_orign.columns)
    x=np.array(x_orign)
    y=np.array(y_orign)
    Rf=RandomForestClassifier(n_estimators=n_iter,random_state=555,max_depth=4)
    skf=StratifiedKFold(n_splits=10)
    score_xy=[]
    for i_train,i_test in skf.split(x,y):
        x_train=x[i_train]
        y_train=y[i_train]
        ros=RandomOverSampler(random_state=11)
        x_res,y_res=ros.fit_sample(x_train,y_train)
        Rf.fit(x_res,y_res)
        score_xy.append(Rf.feature_importances_)
    score_xy=np.array(score_xy).mean(axis=0)
    score_xy=pd.DataFrame(score_xy)
    res=pd.concat([label,score_xy],axis=1)
    res.columns=['label','score']
    res_ed=res.sort_values(by='score',axis=0,ascending=False)
    return res_ed

#6.pearsonr特征选择
from scipy.stats import pearsonr
def feature_value_PS(x,y):#x,y dataframe type
    x_arr=np.array(x)
    y_arr=np.array(y)
    res=[]
    skf=StratifiedKFold(n_splits=10)
    for i_train,i_test in skf.split(x_arr,y_arr):
        res_temp=[]
        x_train=x_arr[i_train]
        y_train=y_arr[i_train]
        res_1=RandomOverSampler(random_state=11)
        x_res,y_res=res_1.fit_sample(x_train,y_train)
        for i in range(x_res.shape[1]):
            temp=pearsonr(x_res[:,i],y_res)
            res_temp.append(temp)
        res.append(res_temp)
    res=np.array(res).mean(axis=0)
    res=pd.DataFrame(res)
    res.columns=['score','p_values']
    res_ed=res.sort_values(by='score',axis=0,ascending=False)
    return res_ed

#显示中文
import matplotlib
zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf',size=16)
#fontproperties=zhfont1

#7.ten classical algorithms
#knn:将每一行test分别与train每一行求距离，找出距离最近的num_k个标签，标签重复最多的为该次test的predict result
#self-code
#from collections import Counter
#def knn(xtrain,xtest,ytrain,num_k):
#    xtrain=np.array(xtrain)
#    xtest=np.array(xtest)
#    size_train=xtrain.shape[0]
#    size_test=xtest.shape[0]
#    ypredict=[]
#    for j in range(size_test):
#        dist=[]
#        for i in range(size_train):
#            diff=xtest[j,:]-xtrain[i,:]
#            dist_i=((diff**2).sum())**0.5
#            dist.append(dist_i)
#        dist_ed=sorted(dist,reverse=True)
#        most_y=[]
#        for k in range(num_k):
#            mostindex=(dist.index(dist_ed[k]))
#            most_y.append(ytrain[mostindex])
#        mosttime_y=Counter(most_y).most_common(1)
#        ypredict.append(int(mosttime_y[0][0]))
#    ypredict=pd.DataFrame(ypredict)
#    return ypredict
#module
from sklearn.neighbors import KNeighborsClassifier as kNN
def kNN_pack(xtrain,xtest,ytrain,k):
    model_kNN=kNN(n_neighbors=k)
    model_kNN.fit(xtrain,ytrain)
    ypre=model_kNN.predict(xtest)
    return ypre
#SVM
from sklearn import svm
def svm_pack(xtrain,xtest,ytrain):
    model=svm.SVC(C=1.0,kernel='linear',gamma=1)
    model.fit(xtrain,ytrain)
    ypre=model.predict(xtest)
    return ypre    
#决策树
from sklearn.tree import DecisionTreeClassifier as DTC
def DTC_pack(xtrain,xtest,ytrain):
    model=DTC(criterion='entropy',random_state=11)
    model.fit(xtrain,ytrain)
    ypre=model.predict(xtest)
    return ypre
#朴素贝叶斯
from sklearn.naive_bayes import GaussianNB as GNB
def bayes_pack(xtrain,xtest,ytrain):
    model=GNB()
    model.fit(xtrain,ytrain)
    ypre=model.predict(xtest)
    return ypre
from sklearn.linear_model import LogisticRegression as LR
def LR_pack(xtrain,xtest,ytrain):
    model=LR()
    model.fit(xtrain,ytrain)
    ypre=model.predict(xtest)
    return ypre
#adaboost
from sklearn.ensemble import AdaBoostClassifier as Ada
def Ada_pack(xtrain,xtest,ytrain):
    model=Ada(n_estimators=100,learning_rate=0.1,random_state=11)
    model.fit(xtrain,ytrain)
    ypre=model.predict(xtest)
    return ypre
#神经网络NeuralNetwork
from sklearn.neural_network import MLPClassifier as MLPC
def MLPC_pack(xtrain,xtest,ytrain):
    model=MLPC(hidden_layer_sizes=(5,2),solver='adam',alpha=1e-5,random_state=11)
    model.fit(xtrain,ytrain)
    ypre=model.predict(xtest)
    return ypre
#bagging
from sklearn.ensemble import BaggingClassifier as BGC
def BGC_pack(xtrain,xtest,ytrain):
    model=BGC(random_state=11)
    model.fit(xtrain,ytrain)
    ypre=model.predict(xtest)
    return ypre
#apriori ?
#result:多分类时需要加以改变
from sklearn.metrics import confusion_matrix,auc,roc_curve,accuracy_score
def result_ml(ytest,testpre):
    res=[]
    TP,FP,FN,TN=np.ravel(confusion_matrix(ytest,testpre))
    ACC=accuracy_score(ytest,testpre)
#    ACC=(TP+TN)/(TP+TN+FP+FN)
    TPR=TP/(TP+FN)
    SPC=TN/(TN+FP)
    PPV=TP/(TP+FP)
    NPV=TN/(TN+FN)
#    FNR=1-TPR
#    FPR=1-SPC
    res=[ACC,TPR,SPC,PPV,NPV]
    return res
    
    
    
