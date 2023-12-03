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


def MFO(pop, dim, lb, ub, MaxIter, fun):
    '''飞蛾扑火优化算法'''
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
    r = 2; #参数
    X = initialization(pop,ub,lb,dim)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitnessS, sortIndex = SortFitness(fitness)  # 对适应度值排序
    Xs = SortPosition(X, sortIndex)  # 种群排序后，初始化火焰位置
    GbestScore = copy.copy(fitnessS[0]) #最优适应度值
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = copy.copy(Xs[0,:])#最优解
    Curve = np.zeros([MaxIter, 1])
    for iter in range(MaxIter):
        print("第"+str(iter)+"次迭代")
        Flame_no=round(pop-iter*((pop-1)/MaxIter)) #火焰数量更新
        r = -1 + iter*(-1)/MaxIter # r 线性从-1降到-2
        #飞蛾扑火行为
        for i in range(pop):
            for j in range(dim):
                if i<= Flame_no:
                    distance_to_flame = np.abs(Xs[i,j] - X[i,j]) #飞蛾与火焰的距离
                    b = 1
                    t = (r - 1)*random.random() + 1          
                    X[i,j] = distance_to_flame*np.exp(b*t)*np.cos(t*2*np.pi) + Xs[i,j] #螺旋飞行
                else:
                    distance_to_flame = np.abs(Xs[Flame_no,j] - X[i,j]) #飞蛾与火焰的距离
                    b = 1
                    t = (r - 1)*random.random() + 1
                    X[i,j] = distance_to_flame*np.exp(b*t)*np.cos(t*2*np.pi) + Xs[Flame_no,j] #螺旋飞行
        
        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测     
        fitness = CaculateFitness(X, fun)  # 计算适应度值
        fitnessS, sortIndex = SortFitness(fitness)  # 对适应度值排序
        Xs = SortPosition(X, sortIndex)  # 种群排序，作为下一代火焰的位置
        if fitnessS[0] <= GbestScore:  # 更新全局最优
            GbestScore = copy.copy(fitnessS[0])
            GbestPositon[0,:] = copy.copy(Xs[0, :])
        Curve[iter] = GbestScore
     
    return GbestScore, GbestPositon, Curve