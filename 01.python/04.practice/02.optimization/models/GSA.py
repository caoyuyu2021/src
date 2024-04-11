import numpy as np
import math
import copy

def initialization(pop,ub,lb,dim):
    ''' 种群初始化函数'''
    '''
    pop:为种群数量
    dim:每个个体的维度
    ub:每个维度的变量上边界，维度为[dim,1]
    lb:为每个维度的变量下边界，维度为[dim,1]
    X:为输出的种群，维度[pop,dim]
    '''
    X = np.zeros([pop,dim]) #声明空间
    for i in range(pop):
        for j in range(dim):
            X[i,j]=(ub[j]-lb[j])*np.random.random()+lb[j] #生成[lb,ub]之间的随机数
    
    return X
     
def BorderCheck(X,ub,lb,pop,dim):
    '''边界检查函数'''
    '''
    dim:为每个个体数据的维度大小
    X:为输入数据，维度为[pop,dim]
    ub:为个体数据上边界，维度为[dim,1]
    lb:为个体数据下边界，维度为[dim,1]
    pop:为种群数量
    '''
    for i in range(pop):
        for j in range(dim):
            if X[i,j]>ub[j]:
                X[i,j] = ub[j]
            elif X[i,j]<lb[j]:
                X[i,j] = lb[j]
    return X


def CaculateFitness(X,fun):
    '''计算种群的所有个体的适应度值'''
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


def SortFitness(Fit):
    '''适应度值排序'''
    '''
    输入为适应度值
    输出为排序后的适应度值，和索引
    '''
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness,index

def SortPosition(X,index):
    '''根据适应度值对位置进行排序'''
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i,:] = X[index[i],:]
    return Xnew

def GSA(pop,dim,lb,ub,MaxIter,fun):
    '''黄金正弦算法'''
    '''
    输入：
    pop:为种群数量
    dim:每个个体的维度
    ub:为个体上边界信息，维度为[1,dim]
    lb:为个体下边界信息，维度为[1,dim]
    fun:为适应度函数接口
    MaxIter:为最大迭代次数
    输出：
    GbestScore:最优解对应的适应度值
    GbestPositon:最优解
    Curve:迭代曲线
    '''
    a = -math.pi
    b = math.pi
    gold = (np.sqrt(5)-1)/2 #黄金分割率
    x1 = a + (1 - gold)*(b-a) #黄金分割系数x1
    x2 = a + gold*(b-a) #黄金分割系数x2  
    X = initialization(pop,ub,lb,dim) #初始化种群
    fitness = CaculateFitness(X,fun) #计算适应度值
    fitness,sortIndex = SortFitness(fitness) #对适应度值排序
    X = SortPosition(X,sortIndex) #种群排序
    GbestScore = copy.copy(fitness[0])
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = copy.copy(X[0,:])#记录最优置
    Curve = np.zeros([MaxIter,1])
    for t in range(MaxIter):
        print('第'+str(t)+'次迭代')
        #根据位置更新公式，更新位置
        for i in range(pop):
            r = np.random.random()
            r1 = 2*math.pi*r
            r2 = r*math.pi
            for j in range(dim):
                X[i,j] = X[i,j]*np.abs(np.sin(r1)) -r2*np.sin(r1)*np.abs(x1*GbestPositon[0,j] - x2*X[i,j])
        
        X = BorderCheck(X,ub,lb,pop,dim) #边界检测  
        fitness = CaculateFitness(X,fun) #计算适应度值
        #更新黄金分割系数
        for i in range(pop):
            if fitness[i]<GbestScore: #如果解优于当前最优解
                GbestScore = fitness[i]
                GbestPositon[0,:] = copy.copy(X[i,:])
                b = x2
                x2 = x1
                x1 = a +(1 - gold)*(b-a)
            else:
                a = x1
                x1 = x2
                x2 = a + gold*(b-a)
            
            if x1 == x2:#如果分割系数相同，随机重置分割系数
                a = -math.pi*np.random.random()
                b = math.pi*np.random.random()
                x1 = a+(1-gold)*(b-a)
                x2 = a+gold*(b-a)
            
        fitness,sortIndex = SortFitness(fitness) #对适应度值排序
        X = SortPosition(X,sortIndex) #种群排序
        if(fitness[0]<=GbestScore): #更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0,:] = copy.copy(X[0,:])
        Curve[t] = GbestScore
    
    return GbestScore,GbestPositon,Curve









