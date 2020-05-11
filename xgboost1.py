#building model XGBOOST and training
import numpy as np
np.random.seed(4)
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
#import lightgbm as lgb
import xgboost as xgb


xtr = pd.read_csv('xtrain.csv')
ytr = pd.read_csv('ytrain.csv')
xval = pd.read_csv('xval.csv')
yval = pd.read_csv('yval.csv')
test = pd.read_csv('ntest.csv')
#print(xtr.info())
#print(ytr.info())
#print(xval.columns)
#print(yval.head)


card_id=test['card_id']
test.drop('card_id',axis=1, inplace=True)

#print(xtr.columns==test.columns,'this is for xtr')
xtr,xte,ytr,yte=train_test_split(xtr,ytr,test_size=0.2,random_state=3)

dtrain = xgb.DMatrix(xtr,label=ytr)
dval = xgb.DMatrix(xval,label=yval)
dtesting = xgb.DMatrix(xte)
dtest = xgb.DMatrix(test)

progress={}
params = {'verbosity':2,'eta':0.05,'eval_metric':'rmse',#'n_estimators':200
'max_depth':5,'min_child_weight':10,
'colsample_bytree':0.7,'subsample':0.6,
'gamma':0.1,
'lambda':30
} #HyperParameters have been set after coarse and fine tuning

num_round =400
xgbo = xgb.train(params,dtrain,num_round,
evals=[(dval,'eval'),(dtrain,'train')],evals_result=progress)
print(progress)
#pdb.set_trace()


#Plotting learning curves

eval_rmse = progress['eval']['rmse']
train_rmse= progress['train']['rmse']
print('train error with std dv:',np.mean(train_rmse),np.std(train_rmse))
print('val error with std:',np.mean(eval_rmse),np.std(eval_rmse))
plt.plot(range(0,num_round), eval_rmse,label='Eval')
plt.plot(range(0,num_round), train_rmse,label='Train')
plt.legend()
plt.show()

#Predicting on created test set 

ypredd = xgbo.predict(dtesting)
mse = mean_squared_error(yte,ypredd)
rmse = np.sqrt(mse)
print('rmse', rmse)

#Predicting on given test set 
ypred = xgbo.predict(dtest)
