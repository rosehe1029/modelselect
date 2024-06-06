#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :model_train.py
@Time         :2024/05/20 09:41:39
@author       :HeRuonan
@version      :1.0
三大基础模型 
'''
import pandas as pd
import os
import matplotlib
import pylab as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['SimHei']  #显示中文
plt.rcParams['axes.unicode_minus']=False
import numpy as np
import joblib
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
path = os.path.dirname(os.path.realpath(__file__)) 


###########场站数据##########
Cap=17
data=pd.read_csv(path+r"/data/with_nwp2024-05-14.csv", parse_dates=True, index_col='时间')
print(data.corr())
data.plot(figsize=(24,10))
plt.savefig(path+r"/pictures/with_nwp.png")
plt.close()
data['实际功率']=data['实际功率'].map(lambda x: x if x> 0 else 0)
data= data[["实际功率","预测风速"]]
#删除某行中某个值为0的行
data= data[data['实际功率'] != np.nan]
data=data.fillna(value='0')

print("*******************"+"短期预测开始"+"**************************")
###########短期预测-1.数据准备##########
train=data['2024-01-21 00:00:00': ]
test=data['2024-05-01 00:15':]
result=pd.DataFrame(columns=['实际功率', '真实值', '预测值'])
result['实际功率']=test['实际功率']['2024-05-01 00:15':]#[96+96-1:]
print(train.shape,test.shape)

x_train, y_train,x_test,y_test=train.iloc[:,1].values.reshape([-1,1]),train.iloc[:,0].values.reshape([-1,1]),test.iloc[:,1].values.reshape([-1,1]),test.iloc[:,0].values.reshape([-1,1])
y_train,y_test=y_train.ravel(),y_test.ravel()
print('x_train.shape',x_train.shape,'y_train.shape',y_train.shape)
print('x_test.shape',x_test.shape,'y_test.shape',y_test.shape)

###########短期预测-2.模型定义##########
model_catboost=CatBoostRegressor(train_dir=path+r'/catboosttrain/',  
                        iterations=200, learning_rate=0.03,
                        depth=6, l2_leaf_reg=3,
                        loss_function='MAE',
                        eval_metric='MAE',
                        random_seed=2013,
                        verbose=0)#参数二model=MultiOutputRegressor(model)

model_lightgbm = LGBMRegressor(objective='regression',verbose=0)

model_ridge=Ridge(alpha=10e-6,fit_intercept=True)

###########短期预测-3.模型训练与预测##########
def shorttime_model_make_result(x_train, y_train,x_test,y_test,model,Cap):
    model.fit(x_train, y_train)
    modelname=type(model).__name__
    print("================="+modelname+"=================")
    #save model
    joblib.dump(model, path+r'/model/'+modelname+r'_shorttime.pkl')
    #load model
    model_= joblib.load(path+r'/model/'+modelname+r'_shorttime.pkl')
    x_test=x_test.reshape(-1,1)
    y_pred = model_.predict(x_test)
    print(y_test.shape,y_pred.shape)
    y_test,y_pred=y_test.ravel(),y_pred.ravel()
    print(y_test.shape,y_pred.shape)

    # 校正
    for j in range(len(y_pred)):
        y_pred[j] = np.round(y_pred[j], 3)
        if y_pred[j] < 0:
            y_pred[j] = float(0)
        if y_pred[j]>Cap:
            y_pred[j]=Cap

    print(y_test.shape,y_pred.shape)
    mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    mae=mean_absolute_error(y_test,y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2score=r2_score(y_test, y_pred)
    print('mse:',mse)
    print('rmse:',rmse)
    print('mae',mae)
    print('mape',mape)
    print('r2score',r2score)

    #分辨率参数-dpi，画布大小参数-figsize
    #plt.figure(dpi=300,figsize=(24,8))
    plt.title(modelname+str("预测结果"))
    plt.plot(y_test,label="真实数据")
    plt.plot(y_pred,label="预测值")
    plt.legend(loc=1)
    plt.savefig(path+r"/pictures/"+modelname+"短期.png")
    plt.close()
    result['真实值']=y_test
    result['预测值']=y_pred
    result.to_csv(path+r"/result/"+modelname+"短期.csv", sep=',')


shorttime_model_make_result(x_train, y_train,x_test,y_test,model_catboost,Cap)
shorttime_model_make_result(x_train, y_train,x_test,y_test,model_lightgbm,Cap)
shorttime_model_make_result(x_train, y_train,x_test,y_test,model_ridge,Cap)


print("******************"+"超短期预测开始"+"**************************")
###########超短期预测-1.数据准备##########
data= data[["实际功率","预测风速"]]
x_=data.iloc[:,1].values.reshape([-1,1])
modelshorttime=joblib.load(path+r'/model/CatBoostRegressor_shorttime.pkl')
data["短期预测值"]=modelshorttime.predict(x_ )

data= data[["实际功率","预测风速","短期预测值",]]
#删除某行中某个值为0的行
data= data[data['实际功率'] != np.nan]
data=data.fillna(value='0')
train_=data['2024-01-21 00:00': ]
test_=data['2024-04-30 16:00':]
result_=pd.DataFrame(columns=['实际功率', '真实值'])
result_['实际功率']=test_['实际功率']['2024-05-01 00:15':]
result16=pd.DataFrame(columns=['实际功率', '真实值'])
result16['实际功率']=test_['实际功率']['2024-05-01 00:15':]
print(train_.shape,test_.shape)

num_nodes=3
def process(dataset):
    #观看过去时间窗口 过去多少天
    past_history_size = 17
    #预测未来值n天
    future_target = 17
    x = []
    y = []
    dataset=dataset.values
    for i in range(len(dataset)-past_history_size-future_target+1):
        x_1=dataset[i:i+past_history_size,0]
        x_2=dataset[i:i+past_history_size,1]
        x_3=dataset[i:i+past_history_size,2]
        x_4=dataset[i+past_history_size:i+past_history_size+future_target,-1]
        x_5=dataset[i+past_history_size:i+past_history_size+future_target,1]
        xxxxx=np.concatenate((x_1,x_2,x_3,x_4,x_5),axis=0) 
        x.append(xxxxx)
        y.append(dataset[i+past_history_size:i+past_history_size+future_target,0])
    x=np.array(x)
    y = np.array(y)   
    return x ,y 

train_x, train_y= process(train_)
test_x,test_y = process(test_)
print(train_x.shape, train_y.shape,test_x.shape,test_y.shape)

###########超短期预测-2.模型定义##########
lightgbm_model= MultiOutputRegressor(LGBMRegressor(objective='regression',verbose=0))

catboost_model=MultiOutputRegressor(CatBoostRegressor(train_dir=path+r'/catboosttrain/',  
                        iterations=200, learning_rate=0.03,
                        depth=6, l2_leaf_reg=3,
                        loss_function='MAE',
                        eval_metric='MAE',
                        random_seed=2013,
                        verbose=0))

ridge_model=Ridge(alpha=10e-6,fit_intercept=True)

###########超短期预测-3.模型训练与预测-输出最后1个点情况##########
def ultrashorttime_model_make_result(train_x, train_y,test_x,test_y,model,modelname,Cap):   
    model.fit(train_x, train_y)
    print("================="+modelname+"=================")
    #save model
    joblib.dump(model, path+r'/model/'+modelname+r'_ultrashorttime.pkl')
    #load model
    model_= joblib.load(path+r'/model/'+modelname+r'_ultrashorttime.pkl')
    pred_y = model_.predict(test_x)
    test_y,pred_y=test_y[:,-1],pred_y[:,-1]
    print(test_y.shape,pred_y.shape)
    mse=mean_squared_error(test_y,pred_y)
    rmse=np.sqrt(mean_squared_error(test_y,pred_y))
    mae=mean_absolute_error(test_y,pred_y)
    mape = mean_absolute_percentage_error(test_y, pred_y)
    r2score=r2_score(test_y, pred_y)
    print('mse:',mse)
    print('rmse:',rmse)
    print('mae',mae)
    print('mape',mape)
    print('r2score',r2score)

    #分辨率参数-dpi，画布大小参数-figsize
    #plt.figure(dpi=300,figsize=(24,8))
    plt.title(modelname+str("预测结果"))
    plt.plot(test_y.ravel(),label="真实数据")
    plt.plot(pred_y.ravel(),label="预测值")
    plt.legend(loc=1)
    plt.savefig(path+r"/pictures/"+modelname+"超短期.png")
    plt.close()
    result_['真实值']=test_y
    result_['预测值']=pred_y
    result_.to_csv(path+r"/result/"+modelname+"超短期.csv", sep=',')


modelname=["Catboost","Lightgbm","Ridge"]
ultrashorttime_model_make_result(train_x, train_y,test_x,test_y,catboost_model,modelname[0],Cap)
ultrashorttime_model_make_result(train_x, train_y,test_x,test_y,lightgbm_model,modelname[1],Cap)
ultrashorttime_model_make_result(train_x, train_y,test_x,test_y,ridge_model,modelname[2],Cap)

###########超短期预测-4.模型训练与预测-输出16个点情况##########
#########按照要求模型预测16个点，但是为了便于部署模型预测出相应的时间点，在实际中，通常多预测出来一个点，变为预测17个点
print("******************"+"超短期16个点预测开始"+"**************************")
def ultrashorttime_model_make_result_16output(train_x, train_y,test_x,test_y,model,modelname,Cap):   
    model.fit(train_x, train_y)
    print("================="+modelname+"=================")
    #save model
    joblib.dump(model, path+r'/model/'+modelname+r'_ultrashorttime.pkl')
    #load model
    model_= joblib.load(path+r'/model/'+modelname+r'_ultrashorttime.pkl')
    pred_y = model_.predict(test_x)
    test_y_,pred_y_=test_y[:,-1],pred_y[:,-1]
    print(test_y_.shape,pred_y_.shape)
    mse=mean_squared_error(test_y_,pred_y_)
    rmse=np.sqrt(mean_squared_error(test_y_,pred_y_))
    mae=mean_absolute_error(test_y_,pred_y_)
    mape = mean_absolute_percentage_error(test_y_, pred_y_)
    r2score=r2_score(test_y_, pred_y_)
    print('mse:',mse)
    print('rmse:',rmse)
    print('mae',mae)
    print('mape',mape)
    print('r2score',r2score)

    #分辨率参数-dpi，画布大小参数-figsize
    #plt.figure(dpi=300,figsize=(24,8))
    plt.title(modelname+str("预测结果"))
    plt.plot(test_y_.ravel(),label="真实数据")
    plt.plot(pred_y_.ravel(),label="预测值")
    plt.legend(loc=1)
    plt.savefig(path+r"/pictures/"+modelname+"超短期17个点取最后一个点画图.png")
    plt.close()
    
    def correction(jj):
        for j in range(len(pred_y[:,jj])):
            pred_y[:,jj][j] = np.round(pred_y[:,jj][j], 3)
            if pred_y[:,jj][j] < 0:
                pred_y[:,jj][j] = float(0)
            if pred_y[:,jj][j]>17:
                pred_y[:,jj][j]=17

    for j in range(16):
        correction(j)
       
    result16['真实值']=y_test
    for i in range(16):
        result16['预测值'+str(i)]=pred_y[:,i]
    
    result16.to_csv(path+r"/result/"+modelname+"超短期16个点.csv", sep=',')


modelname=["Catboost","Lightgbm","Ridge"]
ultrashorttime_model_make_result_16output(train_x, train_y,test_x,test_y,catboost_model,modelname[0],Cap)
ultrashorttime_model_make_result_16output(train_x, train_y,test_x,test_y,lightgbm_model,modelname[1],Cap)
ultrashorttime_model_make_result_16output(train_x, train_y,test_x,test_y,ridge_model,modelname[2],Cap)


