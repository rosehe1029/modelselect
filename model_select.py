#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :model_select.py
@Time         :2024/05/20 16:31:47
@author       :HeRuonan
@version      :1.0
模型筛选
'''
# -*- coding:utf-8 -*-
import os 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
plt.rcParams['font.sans-serif'] = ['SimHei']  #显示中文
plt.rcParams['axes.unicode_minus']=False

###########1.数据准备##########
filename=r'/data/data202401_202404.csv'
path = os.path.dirname(os.path.realpath(__file__)) 
csvpath= os.path.dirname(os.path.realpath(__file__)) + filename
data = pd.read_csv(csvpath, parse_dates=True, index_col='datetime')
data['实际功率']=data['realpower'].map(lambda x: x if x> 0 else 0)
data=data['实际功率']
train,test=data[:'2024/4/1 0:15'],data['2024/4/1 0:15':]

def process(dataset):
    #观看过去时间窗口 过去多少天
    past_history_size = 16
    #预测未来值n天
    future_target = 1
    x = []
    y = []
    dataset=dataset.values
    for i in range(len(dataset)-past_history_size-future_target+1):
        x.append(dataset[i:i+past_history_size])
        y.append(dataset[i+past_history_size:i+past_history_size+future_target])
    x,y= np.array(x),np.array(y)
    x = x.reshape([-1,past_history_size])
    print('*'*100)
    print(x.shape,y.shape)
    return x ,y 

trainX,trainY  = process(train)
testX,testY = process(test)
print(testX.shape,testY.shape)


###########2.模型训练通法##########
def try_different_method(model):
    # 创建对象并填入将样本改成 最高degree次幂
    model.fit(trainX,trainY)
    result = model.predict(testX)
    mse=mean_squared_error(testY,result)
    rmse = np.sqrt(mean_squared_error(testY,result))
    r2score=r2_score(testY,result)
    mae=mean_absolute_error(testY,result)
    mape= mean_absolute_percentage_error(testY,result)
    print('mse',mse)
    print('rmse',rmse)
    print('mae',mae)
    print('mape',mape)
    print('r2score',r2score)

    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.plot(testY.reshape(-1,1),label="真实数据")
    plt.plot(result.reshape(-1,1),label="预测值")
    plt.legend(loc=1)
    plt.title(str(model)[:-2])
    plt.savefig(path+'/pictures/'+str(model)[:-2]+'.png')
    plt.close()
    
    return mse,rmse,mae,mape,r2score
   
    
    

###########3.具体方法选择##########
####3.1决策树回归####
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
####3.2线性回归####
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()
####3.3SVM回归####
from sklearn import svm
model_SVR = svm.SVR()
####3.4KNN回归####
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor(n_neighbors=16)
####3.5随机森林回归####
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=16)#这里使用16个决策树
####3.6Adaboost回归####
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
####3.7GBRT回归####
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
####3.8Bagging回归####
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor()
####3.9ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()
####3.10ARD贝叶斯ARD回归
model_ARDRegression = linear_model.ARDRegression()
####3.11BayesianRidge贝叶斯岭回归
model_BayesianRidge = linear_model.BayesianRidge()
####3.12TheilSen泰尔森估算
model_TheilSenRegressor = linear_model.TheilSenRegressor()
####3.13RANSAC随机抽样一致性算法
model_RANSACRegressor = linear_model.RANSACRegressor()
####3.14Lightgbm梯度提升算法
from lightgbm import LGBMRegressor
model_LGBMRegressor= LGBMRegressor()
####3.15CatBoost梯度提升算法
from catboost import CatBoostRegressor
model_CatBoostRegressor=CatBoostRegressor()


###########4.具体方法调用部分##########
#决策树回归结果
try_different_method(model_DecisionTreeRegressor)
#线性回归结果
#try_different_method(model_LinearRegression)
#SVM回归结果
#try_different_method(model_SVR)
#KNN回归结果
#try_different_method(model_KNeighborsRegressor)
#随机森林回归结果
#try_different_method(model_RandomForestRegressor)
#Adaboost回归结果
#try_different_method(model_AdaBoostRegressor)
#GBRT回归结果
#try_different_method(model_GradientBoostingRegressor)
#Bagging回归结果
#try_different_method(model_BaggingRegressor)
#极端随机树回归结果
#try_different_method(model_ExtraTreeRegressor)
#贝叶斯ARD回归结果
#try_different_method(model_ARDRegression)
#贝叶斯岭回归结果
#try_different_method(model_BayesianRidge)
#泰尔森估算回归结果
#try_different_method(model_TheilSenRegressor)
#随机抽样一致性算法
#try_different_method(model_RANSACRegressor)
#Lightgbm梯度提升算法
#try_different_method(model_LGBMRegressor)
#CatBoost梯度提升算法
#try_different_method(model_CatBoostRegressor)
  
  
###########5.所有方法调用部分##########
method_list=[model_DecisionTreeRegressor,model_LinearRegression,model_SVR,model_KNeighborsRegressor,
             model_RandomForestRegressor,model_AdaBoostRegressor,model_GradientBoostingRegressor,
             model_BaggingRegressor,model_ExtraTreeRegressor,model_ARDRegression,model_BayesianRidge,
             model_TheilSenRegressor,model_RANSACRegressor,model_LGBMRegressor,model_CatBoostRegressor]
result_dict={}
def train_all(method_list,path):
    for method in method_list:
        mse,rmse,mae,mape,r2score=try_different_method(method)
        result_dict[type(method).__name__] = [mse, rmse, mae, mape, r2score]

    df = pd.DataFrame.from_dict(result_dict, orient='index', columns=['mse', 'rmse', 'mae', 'mape', 'r2score'])
    # 将'method'作为行索引
    df.index.name = 'method'
    # 保存DataFrame到CSV
    df.to_csv(f'{path}/result/screen_model_result.csv', encoding='utf-8')


train_all(method_list,path)
#print(result_dict)