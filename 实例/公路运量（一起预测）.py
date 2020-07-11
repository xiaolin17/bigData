import pandas as pd
data=pd.read_excel(r"C:\Users\SUN\Desktop\公路运量.xls",index_col="年份")
train=data.loc[1990:2009].copy()
train_mean=train.mean()
train_std=train.std()
train=(train-train_mean)/train_std
x_train=train[['人数/万人', '机动车数量/万辆', '公路面积/万平方公里']].values
y_train=train[['公路客运量/万人', '公路货运量/万吨']].values

from keras.models import Sequential
from keras.layers.core import Activation,Dense
model=Sequential()
model.add(Dense(input_dim=3,units=8))
model.add(Activation('relu'))
model.add(Dense(input_dim=8,units=2))
model.compile(loss="mean_squared_error",optimizer="adam")
model.fit(x_train,y_train,epochs=5000,batch_size=16)
x=data[['人数/万人', '机动车数量/万辆', '公路面积/万平方公里']]
x=(x-x.mean())/x.std()
a=model.predict(x)
data1=pd.DataFrame(a,columns=['公路客运量/万人', '公路货运量/万吨'],index=range(1990,2012))
data1=data1*train_std[['公路客运量/万人', '公路货运量/万吨']]+train_mean[['公路客运量/万人', '公路货运量/万吨']]
data1.columns=['预测公路客运量/万人', '预测公路货运量/万吨']
data1=data1.round()
data2=pd.concat([data,data1],axis=1)
data2.to_excel(r"C:\Users\SUN\Desktop\预测公路运量.xls")