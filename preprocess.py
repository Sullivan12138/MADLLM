import pandas as pd
import numpy as np
train_new = pd.read_csv('./all_datasets/WADI/WADI.A2_19 Nov 2019/WADI_14days_new.csv')
test_new = pd.read_csv('./all_datasets/WADI/WADI.A2_19 Nov 2019/WADI_attackdataLABLE.csv', skiprows=1)
 
# test = pd.read_csv('D:/anomalydata/wadi/WADI.A1_9 Oct 2017/WADI_attackdata.csv')
# train = pd.read_csv('D:/anomalydata/wadi/WADI.A1_9 Oct 2017/WADI_14days.csv', skiprows=4)
# 这几列都是Nan值，直接赋值0
ncolumns = ['2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS']
train_new[ncolumns]=0
test_new[ncolumns]=0
 
# test_new.columns
# 标签列1为异常-1正常修改为1异常0正常方便后续操作。
test_new.rename(columns={'Attack LABLE (1:No Attack, -1:Attack)':'label'},inplace=True)
test_new.loc[test_new['label'] == 1, 'label'] = 0
test_new.loc[test_new['label'] == -1, 'label'] = 1
 
# 相当于滑动窗口10，stride1的结果，取每个index为10倍数的就是window10，stride10.
train_new_mean = train_new.rolling(10).mean()
train_new_mean = train_new_mean[(train_new_mean.index+1) % 10 ==0]
test_new_mean = test_new.rolling(10).mean()
test_new_mean = test_new_mean[(test_new_mean.index+1) % 10 ==0]
 
# 还有一些Nan值就用上一条数据填充了。
train_new_mean = train_new_mean.fillna(method='ffill')
test_new_mean = test_new_mean.fillna(method='ffill')
 
 
# train_new_mean[train_new_mean.isna().any(axis=1)]
# test_new_mean[test_new_mean.isna().any(axis=1)]
# 取127个特征
wadi_train = train_new_mean.iloc[:,1:].values
wadi_test = test_new_mean.iloc[:,1:-1].values
 
# 标签设置为窗口内的多数，由于取平均，假设10条里大于5条的设置为异常。
f = lambda s: 1 if s>0.5 else 0
 
wadi_labels = test_new_mean['label'].apply(f)

 
# from sklearn.preprocessing import MinMaxScaler
 
# # 最大最小值归一化
# scaler = MinMaxScaler() #实例化
# wadi_train = scaler.fit_transform(wadi_train)
# wadi_test = scaler.fit_transform(wadi_test)
 
np.save('./all_datasets/WADI/wadi_train.npy',wadi_train)
np.save('./all_datasets/WADI/wadi_test.npy',wadi_test)
np.save('./all_datasets/WADI/wadi_labels.npy',wadi_labels)
