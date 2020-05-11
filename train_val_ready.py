# making the  x,y train,val
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import datetime
from sklearn.model_selection import train_test_split
#import lightgbm as lgb
import xgboost as xgb


cnt = pd.read_csv('train.csv')
#mer = pd.read_csv('merchants.csv')
n_mer = pd.read_csv('new_merchant_transactions.csv')
histr = pd.read_csv('historical_transactions.csv')
test = pd.read_csv('ntest.csv')

#pdb.set_trace

a = cnt['first_active_month'].copy()
a = pd.to_datetime(a)

cnt['elapsed_time'] = (datetime.date(2017, 12, 1)- a.dt.date).dt.days

'''
#TRY RUNNING A MODEL HERE 

tr_d = lgb.Dataset(x_tr,y_tr)
params = {
'objective':'regression',
'metrics': 'rsme'
}

lgb_t =lgb.train(params,tr_d)
pred = lgb_t.predict(x_tr)
mse =mean_squared_error(y_tr,pred)
rmse = np.sqrt(mse)
print('train rmse is',rmse)

cv_results = lgb.cv(params, tr_d, nfold=10, metrics = 'rmse',verbose_eval = 10,stratified = False)
print(cv_results) 
'''

#NOW explore other datafiles
#Creating features using historical_transactions and new_merchant_transactions datasets

#Finding a recorded date for each card_id
histr['purchase_date'] = pd.to_datetime(histr['purchase_date'])
a =histr.groupby('card_id').agg({'month_lag':'max', 'purchase_date':'max'}).reset_index()
#print(a.head(10))
a.columns= ['card_id', 'month_lag', 'purchase_date']
f =n_mer.groupby('card_id').agg({'month_lag':'min', 'purchase_date':'min'}).reset_index()
#print(f.head(10))
f.columns= ['card_id', 'month_lag', 'purchase_date']

a['purchase_date']=pd.to_datetime(a['purchase_date'])
a['remov_lags']=a.apply((lambda x: x['purchase_date']-pd.DateOffset(months = x['month_lag'])),axis =1)
t =(a['remov_lags'].dt.day.value_counts())
t_pct = t.div(t.sum(0))
#print(t_pct)
a['obs_date']=a['remov_lags'].dt.to_period('M').dt.to_timestamp()+pd.DateOffset(months= 1)
histr['obs_date']=a['obs_date']


f['purchase_date']=pd.to_datetime(f['purchase_date'])
f['remov_lags']=f.apply((lambda x: x['purchase_date']-pd.DateOffset(months = x['month_lag']-1)),axis =1)
t =(f['remov_lags'].dt.day.value_counts())
t_pct = t.div(t.sum(0))
#print(t_pct)
f['obs_date']=f['remov_lags'].dt.to_period('M').dt.to_timestamp()
#print(type(f),f)
n_mer['obs_date'] = f['obs_date']

#Merging to get obs date for all customers
merg = a.merge(f,how='outer',on='card_id')
med = merg[['card_id','obs_date_x', 'obs_date_y']]
med.columns=['card_id', 'obs_date_his', 'obs_date_n_mer']

med['obs_date_his'].fillna(med['obs_date_n_mer'],inplace=True)
med.drop('obs_date_n_mer',axis =1,inplace =True)
obo =cnt.merge(med,how='left',on='card_id')

#creating other features 
obo['diff']=(obo['obs_date_his']-pd.to_datetime(obo['first_active_month'])).dt.days

obo['obs_date_his'] = pd.to_datetime(obo['obs_date_his'])
obo['n_dff'] = (datetime.date(2018,3,1)-obo['obs_date_his'].dt.date).dt.days

# OHE encoding for 2 variables
for i in ["authorized_flag","category_1"]:
    n_mer[i] = n_mer[i].map({"Y":1,"N":0})
    histr[i] = histr[i].map({"Y":1,"N":0})

#creating features such as min,max,sum,so on, for different card_id/customers    
agg_func = {
    'category_1': ['sum', 'mean'],
    'merchant_id': ['nunique'],
	'state_id':['count','nunique'],
	'subsector_id':['max','count', 'nunique'],
    'month_lag': ['mean', 'max', 'min', 'std'],
    'installments':['count','max'] ,
	'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std'],
	'category_2':['min','max','median'],
	'authorized_flag':['sum', 'mean']
    }
    
agg_histr = histr.groupby(['card_id']).agg(agg_func)
agg_histr.columns = ['_'.join(col).strip() for col in agg_histr.columns.values]
agg_histr.reset_index(inplace=True)
#print (agg_histr)
df1 = (histr.groupby('card_id').size().reset_index(name='transactions_count'))

agg_n_mer =n_mer.groupby(['card_id']).agg(agg_func)
agg_n_mer.columns = ['_'.join(col).strip() for col in agg_n_mer.columns.values]
agg_n_mer.reset_index(inplace=True)
#print (agg_n_mer)
df2 = (n_mer.groupby('card_id').size().reset_index(name='transactions_count'))

obo =obo.merge(df1,how='left',on='card_id')
obo['transactions_count'].fillna(df2['transactions_count'],inplace=True)

agg_histr.fillna(agg_n_mer,inplace=True)
obo = obo.merge(agg_histr,how='left',on='card_id')

obo.fillna(agg_n_mer,inplace=True)
#print(obo.isnull().sum())
'''
#To find the categorical variables which are numerical type 
for i in obo.columns:
 print(i)
 if obo[i].nunique()<10:
  print(obo[i].value_counts())
 else: 
  print(obo[i].nunique())
'''
#print(obo.columns)

#pdb.set_trace()
cat_num_df = ['category_2_median','category_2_max','category_2_min','feature_1','feature_2']
for i in cat_num_df:
 cat_df = pd.get_dummies(obo[i],prefix=i,drop_first=True)
 print(cat_df.head(3))
 obo = pd.concat([obo,cat_df],axis=1)
 obo.drop(i,axis=1,inplace=True)

 
#Creating training and validation sets

tr,val = train_test_split(obo,test_size=0.2,random_state=4)

xtr = tr.drop(['card_id', 'target','first_active_month','obs_date_his'],axis=1)
ytr = tr['target']
xval = val.drop(['card_id', 'target','first_active_month','obs_date_his'],axis=1)
yval = val['target']

#Saving them to be used for training and validation
xtr.to_csv('xtrain.csv',index=False)
ytr.to_csv('ytrain.csv',index=False)
xval.to_csv('xval.csv',index=False)
yval.to_csv('yval.csv',index=False)