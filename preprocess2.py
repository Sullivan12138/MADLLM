import pandas as pd
import numpy as np

train_new = pd.read_csv('./all_datasets/WADI/WADI.A2_19 Nov 2019/WADI_14days_new.csv')
test_new = pd.read_csv('./all_datasets/WADI/WADI.A2_19 Nov 2019/WADI_attackdataLABLE.csv', skiprows=1)
 
 
ncolumns = ['2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS']
train_new = train_new.drop(ncolumns,axis=1)
test_new = test_new.drop(ncolumns,axis=1)
train_new = train_new.dropna(axis=0,how='all')
test_new = test_new.dropna(axis=0,how='all')
test_new = test_new.iloc[:,3:]
train_new = train_new.iloc[:,3:]
 
test_new.rename(columns={'Attack LABLE (1:No Attack, -1:Attack)':'label'},inplace=True)
test_new.loc[test_new['label'] == 1, 'label'] = 0
test_new.loc[test_new['label'] == -1, 'label'] = 1
wadi_labels = test_new['label']
test_new = test_new.iloc[:,:-1]
 
# from sklearn.preprocessing import MinMaxScaler
 
# # 最大最小值归一化
# scaler = MinMaxScaler() #实例化
# wadi_train = scaler.fit_transform(train_new)
# wadi_test = scaler.fit_transform(test_new)
 
np.save('./all_datasets/WADI/wadi_train2.npy',train_new)
np.save('./all_datasets/WADI/wadi_test2.npy',test_new)
np.save('./all_datasets/WADI/wadi_labels2.npy',wadi_labels)