import numpy as np
import random
from sklearn.metrics import accuracy_score
def training():
 # train_datas=[[1, 3, 1], [2, 5, 1], [3, 8, 1], [2, 6, 1],[3, 1, -1], [4, 1, -1], [6, 2, -1], [7, 3, -1]];
 train_datas=[[0.2,0.3,1],[0.3,0.2,1],[0.7,0.5,-1],[0.9,0.7,-1]]
 a=np.array(train_datas)
 a=a[:,2]
 weight=[0,0];#权重
 bias=0;#偏置量
 learning_rate=0.5#学习速率
 i=0;
 train_num=20;
 for i in range(train_num):
  train = random.choice(train_datas)
  x1,x2,y=train;
  predict=np.sign(weight[0]*x1+weight[1]*x2+bias)#输出
  predict=int(predict)
  # print(x1,x2,predict)
  if y*predict<=0:#判断误分类点
   # print("判断错误")
   # print("原来为%f,%f,%f"%(weight[0],weight[1],bias))
   weight[0]=weight[0]+learning_rate*y*x1#更新权重
   weight[1]=weight[1]+learning_rate*y*x2
   bias=bias+learning_rate*y#更新偏置
   # print("修改为%f,%f,%f"%(weight[0],weight[1],bias))
 return weight,bias,a

def testing():
 predict1=[]
 weight,bias,a=training()
 print(weight,bias)
 # test=[[1, 3,1], [2, 5, 1], [3, 8, 1], [2, 6, 1], [3, 1, -1], [4, 1, -1], [6, 2, -1], [7, 3, -1]]
 test=[[0.2,0.3],[0.3,0.2],[0.7,0.5],[0.9,0.7]]
 for i in range(0,len(test)):
  test1=test[i]
  predict1.append(np.sign(weight[0] * test1[0] + weight[1] * test1[1] + bias))
 test_data=[0.7,0.9]
 predict=np.sign(weight[0] * test_data[0] + weight[1] * test_data[1] + bias)
 print(predict)
 score=accuracy_score(a,predict1)
 print(score)
if __name__=='__main__':
 training();
 testing();




