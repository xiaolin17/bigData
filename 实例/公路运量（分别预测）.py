import pandas as pd
data=pd.read_excel(r"C:\Users\SUN\Desktop\公路运量.xls",index_col="年份")
train=data.loc[1990:2009].copy()
train_mean=train.mean()
train_std=train.std()
train=(train-train_mean)/train_std
x_train=train[['人数/万人', '机动车数量/万辆', '公路面积/万平方公里']].values
y_train_K=train['公路客运量/万人'].values
y_train_H=train['公路货运量/万吨'].values

from keras.models import Sequential
from keras.layers.core import Activation,Dense
model_K=Sequential()
model_H=Sequential()
model_K.add(Dense(input_dim=3,units=8))
model_H.add(Dense(input_dim=3,units=8))
model_K.add(Activation('sigmoid'))
model_H.add(Activation('sigmoid'))
model_K.add(Dense(input_dim=8,units=1))
model_H.add(Dense(input_dim=8,units=1))
model_K.compile(loss="mean_squared_error",optimizer="sgd")
model_H.compile(loss="mean_squared_error",optimizer="sgd")
model_K.fit(x_train,y_train_K,epochs=5000,batch_size=16)
model_H.fit(x_train,y_train_H,epochs=5000,batch_size=16)
x=data[['人数/万人', '机动车数量/万辆', '公路面积/万平方公里']]
x=(x-x.mean())/x.std()
a_K=model_K.predict(x)
a_H=model_H.predict(x)
data_K=pd.DataFrame(a_K,columns=['公路客运量/万人'],index=range(1990,2012))
data_H=pd.DataFrame(a_H,columns=['公路货运量/万吨'],index=range(1990,2012))
data_K=data_K*train_std['公路客运量/万人']+train_mean['公路客运量/万人']
data_H=data_H*train_std['公路货运量/万吨']+train_mean['公路货运量/万吨']

data_K.columns=['预测公路客运量/万人']
data_H.columns=['预测公路货运量/万吨']
data_K=data_K.round()
data_H=data_H.round()
data2=pd.concat([data,data_K],axis=1)
data3=pd.concat([data2,data_H],axis=1)
data3.to_excel(r"C:\Users\SUN\Desktop\预测公路运量(分别预测）2.xls")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False
plt.figure()
p_K=data3[['公路客运量/万人','预测公路客运量/万人']].plot(subplots=True,style=['b-o','r-*'])
plt.title("预测公路客运量")
plt.show()
plt.figure()
p_H=data3[['公路货运量/万吨','预测公路货运量/万吨']].plot(subplots=True,style=['b-o','r-*'])
plt.title("预测公路货运量")
plt.show()