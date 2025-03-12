import numpy as np
import random
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


def BOA(pop, dim, lb, ub, MaxIter, fun):
    '''蝴蝶优化算法'''
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
    p=0.8 #切换概率
    power_exponent=0.1  #功率指数a
    sensory_modality=0.1 #感知形态c
    X = initialization(pop,ub,lb,dim)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    indexBest = np.argmin(fitness) #寻找最优适应度位置
    GbestScore = fitness[indexBest] #记录最优适应度值
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = X[indexBest, :]
    X_new = copy.copy(X)
    Curve = np.zeros([MaxIter, 1])
    for t in range(MaxIter):      
        print("第"+str(t)+"次迭代")
        for i in range(pop):
            FP = sensory_modality*(fitness[i]**power_exponent) #刺激强度I的计算
            if random.random()<p: #全局搜索
                dis = random.random()*random.random()*GbestPositon - X[i,:]
                Temp = np.matrix(dis*FP)
                X_new[i,:] = X[i,:] + Temp[0,:]
            else:#局部搜索
                Temp = range(pop)
                JK = random.sample(Temp,pop) #随机选择个体
                dis=random.random()*random.random()*X[JK[0],:]-X[JK[1],:]
                Temp = np.matrix(dis*FP)
                X_new[i,:] = X[i,:] + Temp[0,:]
            for j in range(dim):
                if X_new[i,j] > ub[j]:
                    X_new[i, j] = ub[j]
                if X_new[i,j] < lb[j]:
                    X_new[i, j] = lb[j]
            
            #如果更优才更新
            if(fun(X_new[i,:])<fitness[i]):
                X[i,:] = copy.copy(X_new[i,:])
                fitness[i] = copy.copy(fun(X_new[i,:]))
        
        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测
        fitness = CaculateFitness(X, fun)  # 计算适应度值
        indexBest = np.argmin(fitness)
        if fitness[indexBest] <= GbestScore:  # 更新全局最优
            GbestScore = copy.copy(fitness[indexBest])
            GbestPositon[0,:] = copy.copy(X[indexBest, :])
        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve
