import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn import metrics

from lightgbm import LGBMRegressor, LGBMClassifier

from tqdm import tqdm
from sklearn.linear_model import LinearRegression as lr
from sklearn.tree import DecisionTreeRegressor as dt_r
from sklearn.ensemble import RandomForestRegressor as rf_r
import time
from sklearn.metrics import f1_score, roc_auc_score, classification_report, mean_absolute_error, r2_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 데이터 로드
train=pd.read_csv('data/train.csv', index_col='id')
test=pd.read_csv('data/test.csv', index_col='id')
submission=pd.read_csv('data/sample_submission.csv', index_col='id')

# src, dst, target list
src_list = ['650_src', '660_src', '670_src', '680_src', '690_src', '700_src',
       '710_src', '720_src', '730_src', '740_src', '750_src', '760_src',
       '770_src', '780_src', '790_src', '800_src', '810_src', '820_src',
       '830_src', '840_src', '850_src', '860_src', '870_src', '880_src',
       '890_src', '900_src', '910_src', '920_src', '930_src', '940_src',
       '950_src', '960_src', '970_src', '980_src', '990_src']
dst_list = ['650_dst',
       '660_dst', '670_dst', '680_dst', '690_dst', '700_dst', '710_dst',
       '720_dst', '730_dst', '740_dst', '750_dst', '760_dst', '770_dst',
       '780_dst', '790_dst', '800_dst', '810_dst', '820_dst', '830_dst',
       '840_dst', '850_dst', '860_dst', '870_dst', '880_dst', '890_dst',
       '900_dst', '910_dst', '920_dst', '930_dst', '940_dst', '950_dst',
       '960_dst', '970_dst', '980_dst', '990_dst']
target_list = ['hhb', 'hbo2', 'ca', 'na']

# 결측치 보간
train[dst_list] = train[dst_list].interpolate(axis=1)
test[dst_list] = test[dst_list].interpolate(axis=1)

# rho 변경
train['rho'] = train['rho']/5
test['rho'] = test['rho']/5

# bfill , ffill
train[dst_list] = train[dst_list].fillna(method = 'bfill',axis=1)
test[dst_list] = test[dst_list].fillna(method = 'bfill',axis=1)


# train.loc[train['650_dst']==0, ['650_dst','660_dst','670_dst']]
# train.iloc[6,1:36].plot()
# train.iloc[6,36:71].plot()


# 0 to min
train[src_list] = train[src_list].replace(0, np.NaN)
train[dst_list] = train[dst_list].replace(0, np.NaN)
test[src_list] = test[src_list].replace(0, np.NaN)
test[dst_list] = test[dst_list].replace(0, np.NaN)

train[src_list] = train[src_list].apply(lambda x: x.fillna(train[src_list].min(1)), axis=0)
train[dst_list] = train[dst_list].apply(lambda x: x.fillna(train[dst_list].min(1)), axis=0)
test[src_list] = test[src_list].apply(lambda x: x.fillna(test[src_list].min(1)), axis=0)
test[dst_list] = test[dst_list].apply(lambda x: x.fillna(test[dst_list].min(1)), axis=0)

###비율변수추가
log_col = []
for i in range(650,1000,10):
    log_col.append('src'+str(i)+'dst')

df_tr_log = pd.DataFrame(np.log(train[src_list].values)/np.log(train[dst_list].values),columns=log_col)
df_te_log = pd.DataFrame(np.log(test[src_list].values)/np.log(test[dst_list].values),columns=log_col)


tr_log = pd.DataFrame(np.log(train[src_list].values/train[dst_list].values),columns=log_col)
te_log = pd.DataFrame(np.log(test[src_list].values/test[dst_list].values),columns=log_col,index=submission.index)


# train_test_split
train_X = pd.concat((train['rho'],np.log(train[src_list+dst_list])),axis=1)
train_Y = train[target_list]
train_x, test_x, train_y, test_y = train_test_split(train_X,train_Y,random_state=0)
test_X = pd.concat((test['rho'],np.log(test[src_list+dst_list])),axis=1)

# modeling
# params = {'learning-rate' : 0.01,
#            'max_depth' : 16,
#            'boosting_type' : 'gbdt',
#            'objective' : 'regression',
#            'metric' : 'mae',
#            'is_training_metric' : True,
#            'nem_leaves' : 144,
#            'feature_fracton' : 0.9,
#            'bagginf_fraction' : 0.7,
#            'bagginf_freq' : 5,
#            'silent' : False}
# model = MultiOutputRegressor(LGBMRegressor(**params, random_state = 0), n_jobs = -1)
# model.fit(train_x, train_y)

# mean_absolute_error(model.predict(test_x), test_y) # 1.14

# def model_scoring_cv(model, x, y, cv=10):
#     start=time.time()
#     score=-cross_val_score(model, x, y, cv=cv, scoring='neg_mean_absolute_error').mean()
#     stop=time.time()
#     print(f"Validation Time : {round(stop-start, 3)} sec")
#     return score

# model_scoring_cv(model, train_X, train_Y) #1.14 56sec

# keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

model2 = Sequential()
model2.add(Dense(train_x.shape[1], input_dim = train_x.shape[1], activation = 'relu'))
model2.add(Dense(35, activation='relu'))
model2.add(Dense(4))

model2.compile(loss = 'mean_absolute_error',
             optimizer = 'adam',
             metrics = ['accuracy','mae'])

start = time.time()
model2.fit(train_x,train_y,epochs = 1333, batch_size = 80)
stop = time.time()
print('\n')
print(f'time : {int((stop - start)//60)} min {round(((stop - start)%60)*60/100)} sec')

# evaluate
model2.evaluate(train_x, train_y) # 1.14 #1.08(16 레이어추가) # 1.10(8레이어) # 1.05(16,16,16) #0.71
#35(rho, logsrc, logdst)1.01

model2.predict(test_x)
mean_absolute_error(test_y.values, model2.predict(test_x)) #1.16343107116717 / 제출1.19
#71레이어 - 1.14
#35(rho, logsrc, logdst,logsrc/logdst - 1.11/ 제출-1.14
#35(rho, logsrc, logdst) - 1.14/

# 16레이어 - 1.17
# 8 레이어 -1.17
# 161616, 1.14
model2.predict(test_X).min()

#submission
pred = model2.predict(test_X)
preds = pd.DataFrame(data = pred, columns = submission.columns, index= submission.index)
preds.to_csv('ANN_log35.csv')
pd.read_csv('ANN_log35.csv')
pd.read_csv('ANN_log35.csv').min()

model.predict(test_X)


