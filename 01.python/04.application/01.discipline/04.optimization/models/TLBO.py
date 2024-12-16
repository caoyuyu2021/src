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


def TLBO(pop, dim, lb, ub, MaxIter, fun):
    '''教与学优化算法'''
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

    X = initialization(pop,ub,lb,dim)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    GbestScore = np.min(fitness) #寻找最优适应度值
    indexBest = np.argmin(fitness) #最优适应度值对应得索引
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = copy.copy(X[indexBest, :])#记录最优解
    Curve = np.zeros([MaxIter, 1])
    for t in range(MaxIter):
        print('第'+str(t)+'次迭代')
        for i in range(pop):
            #教阶段
            Xmean = np.mean(X) #计算平均位置
            indexBest = np.argmin(fitness) #寻找最优位置      
            Xteacher = copy.copy(X[indexBest,:]) #老师的位置，即最优位置
            beta = random.randint(0,1)#教学因子
            Xnew = X[i,:] + np.random.random(dim)*(Xteacher - beta*Xmean) #教阶段位置更新
            #边界检查
            for j in range(dim):
                if Xnew[j]>ub[j]:
                    Xnew[j] = ub[j]
                if Xnew[j]<lb[j]:
                    Xnew[j]=lb[j]      
            #计算新位置适应度
            fitnessNew = fun(Xnew);
            #如果新位置更优，则更新先前解
            if fitnessNew<fitness[i]:
                X[i,:] = copy.copy(Xnew)
                fitness[i] = copy.copy(fitnessNew)
            #学阶段
            p = random.randint(0,dim-1)#随机选择一个索引
            while i == p:#确保随机选择的索引不等于当前索引
                p = random.randint(0,dim-1)
            #学阶段位置更新
            if fitness[i]<fitness[p]:
                Xnew = X[i,:] + np.random.random(dim)*(X[i,:] - X[p,:])
            else:
                Xnew = X[i,:] - np.random.random(dim)*(X[i,:] - X[p,:])
            #边界检查
            for j in range(dim):
                if Xnew[j]>ub[j]:
                    Xnew[j] = ub[j]
                if Xnew[j]<lb[j]:
                    Xnew[j]=lb[j]
            #如果新位置更优，则更新先前解
            fitnessNew = fun(Xnew)
            #如果新位置更优，则更新先前解
            if fitnessNew<fitness[i]:
                X[i,:] = copy.copy(Xnew)
                fitness[i] = fitnessNew
                             
        fitness = CaculateFitness(X, fun)  # 计算适应度值
        indexBest = np.argmin(fitness)
        if fitness[indexBest] <= GbestScore:  # 更新全局最优
            GbestScore = copy.copy(fitness[indexBest])
            GbestPositon[0,:] = copy.copy(X[indexBest, :])
        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve

