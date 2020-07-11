import numpy as np
import random

def goldsteinsearch(f,df,d,x,alpham,rho,t):

  flag=0

  a=0
  b=alpham
  fk=f(x)
  gk=df(x)

  phi0=fk
  dphi0=np.dot(gk,d)

  alpha=b*random.uniform(0,1)

  while(flag==0):
    newfk=f(x+alpha*d)
    phi=newfk
    if(phi-phi0<=rho*alpha*dphi0):
      if(phi-phi0>=(1-rho)*alpha*dphi0):
        flag=1
      else:
        a=alpha
        b=b
        if(b<alpham):
          alpha=(a+b)/2
        else:
          alpha=t*alpha
    else:
      a=a
      b=alpha
      alpha=(a+b)/2
  return alpha

import matplotlib.pyplot as plt
def rosenbrock(x):
  return x[0]**2+25*x[1]**2

def jacobian(x):
  return np.array([2*x[0],50*x[1]])


X1=np.arange(-6,6+0.01,0.01)
X2=np.arange(-6,6+0.01,0.01)
[x1,x2]=np.meshgrid(X1,X2)
f=x1**2+25*x2**2; # 给定的函数
plt.contour(x1,x2,f,10) # 画出函数的20条轮廓线

def steepest(x0):

  print('初始点为:')
  print(x0,'\n')
  imax = 200
  W=np.zeros((2,imax))
  W[:,0] = x0
  i = 1
  x = x0
  grad = jacobian(x)
  delta = sum(grad**2) # 初始误差


  while i<imax and delta>10**(-5):
    p = -jacobian(x)
    x0=x
    alpha = goldsteinsearch(rosenbrock,jacobian,p,x,1,0.1,2)*0.1
    x = x + alpha*p
    W[:,i] = x
    grad = jacobian(x)
    delta = sum(grad**2)
    i=i+1

  print("迭代次数为:",i)
  print("近似最优解为:")
  print(x,'\n')
  W=W[:,0:i] # 记录迭代点
  return W

x0 = np.array([2,2])
W=steepest(x0)

plt.plot(W[0,:],W[1,:],'g*',W[0,:],W[1,:]) # 画出迭代点收敛的轨迹
plt.show()