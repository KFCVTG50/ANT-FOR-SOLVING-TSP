from numpy import *
from random import *
from math import *
from matplotlib.pyplot import *
import time
start = time.time()  # 时间计时器

# 读取TSP数据
def read_tsp_file(file_path):
    cities = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'NODE_COORD_SECTION' in line:
                for j in range(i+1, len(lines)):
                    if 'EOF' in lines[j]:
                        break
                    x, y = lines[j].split()[1:]
                    cities.append((float(x), float(y)))
                break
    return array(cities)

# 计算各城市之间的距离，得到距离矩阵
def DistMat(n, cities):  # n为城市个数， cities为城市坐标
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist[i][j] = dist[j][i] = round(np.linalg.norm(cities[i] - cities[j])) # linalg.norm 求向量的范数（默认求 二范数），得到 i、j 间的距离
    return dist

# 计算TSP路径长度
def RouteLength(route, dist):
    n = len(route)
    L = dist[route[n-1], route[0]]
    for i in range(n - 1):
        L += dist[route[i], route[i + 1]]
    return round(L)

# 交换算子
def Swap(route):
    n = len(route)
    i, j = sample(range(n), 2)  # 在 [0,n) 产生 2个不相等的随机整数 i,j
    route1 = route.copy()
    route1[i], route1[j] = route[j], route[i]
    return route1

# 插入算子
def Insertion(route):
    """把第二个索引数字插到第一个索引后边"""
    n = len(route)
    i, j = sample(range(n), 2)
    # s = min([i, j])
    # k = max([i, j])
    gap = route[j]
    data = np.delete(route, [j])
    route1 = np.insert(data, i, gap)  # 是将gap插入到data列表中第s+1索引上
    return route1

# 反转算子
def Reversion(route):
    """随机选两个索引下标，把两个索引之间(含两个索引)的数组反转"""
    n = len(route)
    route1 = route.copy()
    i, j = sample(range(n), 2)
    s = min([i, j])
    k = max([i, j])
    gap = route[s:k+1]
    gap1 = gap[::-1]  # 片段反转
    route1[s:k+1] = gap1
    return route1

# 轮盘赌选择
def Roulette(pSwap, pInsertion, pReversion):
    list1 = [pSwap, pInsertion, pReversion]
    sum_list1 = sum(list1)
    rand_num = random()
    probability = 0
    for i in range(len(list1)):
        probability += list1[i]/sum_list1  # 概率累加
        if probability >= rand_num:
            return i

# 邻域方式选择
def Neighbor(route1, pSwap, pInsertion, pReversion):
    index = Roulette(pSwap, pInsertion, pReversion)
    if index == 0:
        route2 = Swap(route1)
    elif index == 1:
        route2 = Insertion(route1)
    else:
        route2 = Reversion(route1)
    return route2

# 绘制路径图
def plot_tsp_path(cities, path):
    plot(*zip(*cities), 'co')
    for i in range(len(path) - 1):
        x, y = zip(cities[path[i]], cities[path[i+1]])
        plot(x, y, 'b')
    x, y = zip(cities[path[-1]], cities[path[0]])
    for i, city in enumerate(cities):  # 在路径图上加序号
        annotate(str(i), (city[0], city[1]))
    plot(x, y, 'b')
    xlabel('x')
    ylabel('y')
    show()

# 参数
MaxOutIter = 1000  # 外循环次数
MaxInIter = 500  # 内循环次数
T0 = 1000  # 初始温度
alpha = 0.99  # 冷却因子
pSwap = 0.2  # 交换概率
pInsertion = 0.3  # 插入变换概率
pReversion = 0.5  # 反转概率

cities = read_tsp_file('eil51.tsp')  # 添加城市坐标数据
n = len(cities)
dist = DistMat(n, cities)  # 距离矩阵表
# 构造初始解
# currRoute = arange(n)  # 随机生成初始解
# shuffle(currRoute)  # 将列表打乱
currRoute = np.random.permutation(n)
currL = RouteLength(currRoute, dist)  # 初始解总距离
bestRoute = currRoute.copy()  # 初始将初始解赋值给全局最优解
bestL = currL
T = T0
recordBest = []
# 主循环
for outIter in range(1, MaxOutIter+1):
    for inIter in range(1, MaxInIter+1):
        newRoute = Neighbor(currRoute, pSwap, pInsertion, pReversion)  # 经过邻域结构后产生的新的路线
        # newRoute = Insertion(currRoute)

        newL = RouteLength(newRoute, dist)
        # 如果新路线比当前路线更好，则更新当前路线，以及当前路线总距离
        if newL <= currL:
            currRoute = newRoute
            currL = newL
        else:
            # 如果新路线不如当前路线好，则采用退火准则，以一定概率接受新路线
            delta = (newL - currL)  # 计算新路线与当前路线总距离之差
            P = exp(-delta/T)  # 计算接受新路线的概率
            if random() <= P:
                currRoute = newRoute
                currL = newL
        if currL <= bestL:
            bestRoute = currRoute.copy()
            bestL = currL
    print(f'第{outIter}次迭代：全局最优路线总距离= {bestL}')
    recordBest.append(bestL)
    T = alpha * T
end = time.time()
print(f'CPU执行时间:{end - start} 秒',)

figure1 = figure()  # 创建图形窗口 2
plot(array(recordBest), 'b-', label='Best')

figure2 = figure()  # 创建图形窗口 1
plot_tsp_path(cities, bestRoute)



