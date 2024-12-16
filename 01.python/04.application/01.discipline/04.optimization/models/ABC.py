import numpy as np
import copy as copy

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
    '''适应度排序'''
    '''
    输入为适应度值
    输出为排序后的适应度值，和索引
    '''
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness,index

def SortPosition(X,index):
    '''根据适应度对位置进行排序'''
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i,:] = X[index[i],:]
    return Xnew


def RouletteWheelSelection(P):
    '''轮盘赌策略'''
    C = np.cumsum(P)#累加
    r = np.random.random()*C[-1]#定义选择阈值，将随机概率与总和的乘积作为阈值
    out = 0
    #若大于或等于阈值，则输出当前索引，并将其作为结果，循环结束
    for i in range(P.shape[0]):
        if r<C[i]:
            out = i
            break
    return out
        

def ABC(pop, dim, lb, ub, MaxIter, fun):
    '''人工蜂群算法'''
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
    L = round(0.6*dim*pop) #limit 参数
    C = np.zeros([pop,1]) #计数器，用于与limit进行比较判定接下来的操作
    nOnlooker=pop #引领蜂数量
    
    X= initialization(pop,ub,lb,dim)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = copy.copy(fitness[0])#记录最优适应度值
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = copy.copy(X[0, :])#记录最优位置
    Curve = np.zeros([MaxIter, 1])
    Xnew = np.zeros([pop,dim])
    fitnessNew = copy.copy(fitness)
    for t in range(MaxIter):
        '''引领蜂搜索'''
        for i in range(pop):
            k = np.random.randint(pop)#随机选择一个个体
            while(k==i):#当k=i时，再次随机选择，直到k不等于i
                k = np.random.randint(pop)
            phi = (2*np.random.random([1,dim]) - 1)
            Xnew[i,:] = X[i,:]+phi*(X[i,:]-X[k,:]) #公式(2.2)位置更新
        Xnew=BorderCheck(Xnew,ub,lb,pop,dim) #边界检查
        fitnessNew = CaculateFitness(Xnew, fun)  # 计算适应度值
        for i in range(pop):
            if fitnessNew[i]<fitness[i]:#如果适应度值更优，替换原始位置
                X[i,:]= copy.copy(Xnew[i,:])
                fitness[i] = copy.copy(fitnessNew[i])
            else:
                C[i] = C[i]+1 #如果位置没有更新，累加器+1
                
        #计算选择适应度权重
        F = np.zeros([pop,1])
        MeanCost = np.mean(fitness)
        for i in range(pop):
            F[i]=np.exp(-fitness[i]/MeanCost)
        P=F/sum(F) #式（2.4）
        '''侦察蜂搜索'''
        for m in range(nOnlooker):
            i=RouletteWheelSelection(P)#轮盘赌测量选择个体
            k = np.random.randint(pop)#随机选择个体
            while(k==i):
                k = np.random.randint(pop)
            phi = (2*np.random.random([1,dim]) - 1)
            Xnew[i,:] = X[i,:]+phi*(X[i,:]-X[k,:])#位置更新
        Xnew=BorderCheck(Xnew,ub,lb,pop,dim)#边界检查
        fitnessNew = CaculateFitness(Xnew, fun)  # 计算适应度值
        for i in range(pop):
            if fitnessNew[i]<fitness[i]:#如果适应度值更优，替换原始位置
                X[i,:]= copy.copy(Xnew[i,:])
                fitness[i] = copy.copy(fitnessNew[i])
            else:
                C[i] = C[i]+1 #如果位置没有更新，累加器+1
        '''判断limit条件，并进行更新'''
        for i in range(pop):
            if C[i]>=L:
                for j in range(dim):
                    X[i, j] = np.random.random() * (ub[j] - lb[j]) + lb[j]
                    C[i] = 0
                
        fitness = CaculateFitness(X, fun)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        if fitness[0] <= GbestScore:  # 更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0,:] = copy.copy(X[0, :])
        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve