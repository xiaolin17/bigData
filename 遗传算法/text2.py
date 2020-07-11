import numpy as np
import math
import time

class Ga():

    def __init__(self):
        self.popsize=500    #初始种群个数
        self.pc=0.9 #交配概率
        self.pm=0.01    #变异概率
        self.precision=0.0001   #运算精度
        self.boundsend=2    #x的取值范围
        self.boundsbegin=-1 #x的取值范围
        self.Bitlength=int(np.ceil(np.log2((self.boundsend-self.boundsbegin)/self.precision)))  #二进制长度16位
        self.population = np.random.randint(0, 2, size=(self.popsize, self.Bitlength))  #初始化

    """计算出适应度"""

    def fitness(self, population):
        Fitvalue = []
        for i in population:
            x = self.transform2to10(i)
            xx = self.boundsbegin + x * (self.boundsend - self.boundsbegin) / (math.pow(2, self.Bitlength) - 1)
            s = self.targetfun(xx)
            s=s+230
            Fitvalue.append(s)
        fsum = sum(Fitvalue)
        everypopulation = [x / fsum for x in Fitvalue]

        cumsump = np.cumsum(everypopulation).tolist()

        return Fitvalue, cumsump

    """二进制转换为十进制"""

    def transform2to10(self,population):
        x=0
        n=self.Bitlength
        p=population.copy()
        p=p.tolist()
        p.reverse()
        for j in range(n):
            x=x+p[j]*pow(2,j)
        return x

    """选择两个交叉"""

    def select(self,cumsump):
        seln=[]
        for i in range(2):
            j=0
            r=np.random.uniform(0,1)
            prand=[x-r for x in cumsump]
            while prand[j]<0:
                j=j+1
            seln.append(j)
        return seln

    """交叉"""

    def crossover(self,seln,pc):
        d=self.population[seln[1]].copy()
        f=self.population[seln[0]].copy()
        r=np.random.uniform(0,1)
        if r<pc:
            # print(pc)
            # print('yes')
            c=np.random.randint(1,self.Bitlength-1)
            # print(c)
            a=self.population[seln[1]][c:]
            b=self.population[seln[0]][c:]
            d[c:]=b
            f[c:]=a
            # print(d)
            # print(f)
            g=d
            h=f
        else:
            g=self.population[seln[1]]
            h=self.population[seln[0]]
        return g,h

    """变异"""

    def mutation(self,scenw,pm):
        r=np.random.uniform(0,1)
        if r<pm:
            v=np.random.randint(0,self.Bitlength)

            scenw[v]=np.abs(scenw[v]-1)
        else:
            scenw=scenw
        return scenw

    def targetfun(self,x):
        y=x*math.sin(10*math.pi*x)+2                        #目标函数
        return y

def tobinarystring(numerical):
    numa = math.floor(numerical)
    numb = numerical - numa
    bina = bin(numa)
    bina = bina[2:]
    result = "0"*(9-len(bina))
    result += bina
    for i in range(7):
        numb *= 2
        result += str(math.floor(numb))
        numb = numb - math.floor(numb)
    result=[int(x) for x in result]
    result=np.array(result)
    return result


import matplotlib.pyplot as plt

if __name__=="__main__":
    start=time.time()
    Generationmax=100
    gg=Ga()
    # print(gg.population)
    Fitvalue,cumsump=gg.fitness(gg.population)
    # print(Fitvalue)
    # print(cumsump)
    Generation=1
    scnew=[]
    ymax=[]
    ymean=[]
    while Generation<Generationmax+1:
        Fitvalue, cumsump = gg.fitness(gg.population)
        for i in range(0,gg.popsize,2):
            # print("Generation:%d",Generation)
            # print("i:%d",i)
            seln=gg.select(cumsump)
            scro=gg.crossover(seln,gg.pc)
            s1=gg.mutation(scro[0],gg.pm)
            s2=gg.mutation(scro[1],gg.pm)
            scnew.append(s1)
            scnew.append(s2)
        gg.population=scnew
        Fitvalue,cumsump=gg.fitness(gg.population)
        fmax=max(Fitvalue)
        fmean=np.mean(Fitvalue)
        x_mean=tobinarystring(fmean)
        # print(x_mean)
        # print(type(x_mean))
        d=int(Fitvalue.index(fmax))
        # d_mean=int(Fitvalue.index(fmean))
        x=gg.transform2to10(gg.population[d])
        x_mean=gg.transform2to10(x_mean)
        # print(x_mean)
        xx=gg.boundsbegin+x*(gg.boundsend-gg.boundsbegin)/(math.pow(2,gg.Bitlength)-1)
        xx_mean=gg.boundsbegin+x_mean*(gg.boundsend-gg.boundsbegin)/(math.pow(2,gg.Bitlength)-1)
        Generation+=1
        ymax.append(xx)
        ymean.append(xx_mean)
    print(ymax)
    print(ymean)
    
    print("Ymax:")
    print(ymax[-1])
    print("Ymean:")
    print(ymean[-1])
    plt.figure()
    plt.plot(list(np.arange(1,Generationmax+1)),list(np.array(ymax)*10),'r-',label='ymax')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(list(np.arange(1,Generationmax+1)),ymean,'*-',label='ymean')
    plt.legend()
    plt.show()

    plt.figure()
    x=np.linspace(-2,2,100)
    y=200*np.exp(-0.05*x)*np.sin(x)
    plt.plot(x,y,'b--',label='orginal')
    plt.legend()
    plt.show()
    end=time.time()-start















