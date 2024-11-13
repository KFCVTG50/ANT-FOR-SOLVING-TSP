import numpy as np
import matplotlib.pyplot as plt
import random
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑字体
plt.rcParams['axes.unicode_minus'] = False  # 处理负号显示异常

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
    return np.array(cities)

# 计算距离矩阵
def getdistmat(coordinates):
    num = len(coordinates)
    distmat = np.zeros((num, num))
    for i in range(num):
        for j in range(i, num):
            distmat[i][j] = distmat[j][i] = np.linalg.norm(
                coordinates[i] - coordinates[j])
    return distmat

# 绘制路径图
def plot_tsp_path(cities, path):
    plt.plot(*zip(*cities), 'co')
    for i in range(len(path) - 1):
        x, y = zip(cities[path[i]], cities[path[i+1]])
        plt.plot(x, y, 'b')
    x, y = zip(cities[path[-1]], cities[path[0]])
    for i, city in enumerate(cities):  # 在路径图上加序号
        plt.annotate(str(i), (city[0], city[1]))
    plt.plot(x, y, 'b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

coordinates = read_tsp_file('ANT/eil76.tsp')  # 添加城市坐标数据
# 初始化
distmat = getdistmat(coordinates)
numant = 30  # 蚂蚁个数
numcity = len(coordinates)  # 城市个数
alpha = 1  # 信息素重要程度因子
beta = 5  # 启发函数重要程度因子
rho = 0.5  # 信息素的挥发速度
Q = 1  # 信息素释放总量
iter = 0  # 循环次数
itermax = 1000  # 循环最大值
etatable = 1.0 / (distmat + np.diag([1e10] * numcity))  # 启发函数矩阵，表示蚂蚁从城市i转移到矩阵j的期望程度
pheromonetable = np.ones((numcity, numcity))  # 信息素矩阵
pathtable = np.zeros((numant, numcity)).astype(int)  # 路径记录表
distmat = getdistmat(coordinates)  # 城市的距离矩阵
lengthaver = np.zeros(itermax)  # 各代路径的平均长度
lengthbest = np.zeros(itermax)  # 各代及其之前遇到的最佳路径长度
pathbest = np.zeros((itermax, numcity)) # 各代及其之前遇到的最佳路径长度
historyfitness_best = float('inf')
historypath_best = []
historyfitness_table = []
#//核心点-循环迭代
while iter < itermax:
    random_integers = [random.randint(0, numcity-1) for _ in range(numant)]
    for i in range(pathtable.shape[0]):
        pathtable[i, 0] = random_integers[i]
    length = np.zeros(numant)  # 计算各个蚂蚁的路径距离
    for i in range(numant):
        visiting = pathtable[i, 0]  # 当前所在的城市
        unvisited = set(range(numcity))  # 未访问的城市,以集合的形式存储{}
        unvisited.remove(visiting)  # 删除元素；利用集合的remove方法删除存储的数据内容
        for j in range(1, numcity):  # 循环numcity-1次，访问剩余的numcity-1个城市
            # 每次用轮盘法选择下一个要访问的城市
            listunvisited = list(unvisited)
            probtrans = np.zeros(len(listunvisited))
            for k in range(len(listunvisited)):
                probtrans[k] = np.power(pheromonetable[visiting][listunvisited[k]], alpha) \
                    * np.power(etatable[visiting][listunvisited[k]], beta)
            if probtrans.sum() == 0:
                # 如果所有概率为零，随机选择一个未访问的城市
                k = random.choice(listunvisited)
            else:
                cumsumprobtrans = (probtrans / probtrans.sum()).cumsum()
                cumsumprobtrans -= np.random.rand()
                k = listunvisited[(np.where(cumsumprobtrans > 0)[0])[0]]
            # 元素的提取（也就是下一轮选的城市）
            pathtable[i, j] = k  # 添加到路径表中（也就是蚂蚁走过的路径)
            unvisited.remove(k)  # 然后在为访问城市set中remove（）删除掉该城市
            length[i] += distmat[visiting][k]
            visiting = k
        # 蚂蚁的路径距离包括最后一个城市和第一个城市的距离
        length[i] += distmat[visiting][pathtable[i, 0]]
        # 包含所有蚂蚁的一个迭代结束后，统计本次迭代的若干统计参数

    lengthbest = min(length)  # 当前种群最优值
    best_index = np.argmin(length)
    pathbest = pathtable[best_index]  # 当前种群最优路径
    # 历史最优记录
    if lengthbest < historyfitness_best:
        historyfitness_best = lengthbest
        historypath_best = pathbest  # 更新历史最优路径
    historyfitness_table = np.concatenate((historyfitness_table, np.array([historyfitness_best])))  # 存储历史最优值

    # 更新信息素
    changepheromonetable = np.zeros((numcity, numcity))
    for i in range(numant):
        for j in range(numcity - 1):
            changepheromonetable[pathtable[i, j]][pathtable[i, j + 1]] += Q / distmat[pathtable[i, j]][
                pathtable[i, j + 1]]  # 计算信息素增量
        changepheromonetable[pathtable[i, j + 1]][pathtable[i, 0]] += Q / distmat[pathtable[i, j + 1]][pathtable[i, 0]]
    pheromonetable = (1 - rho) * pheromonetable + \
        changepheromonetable  # 计算信息素公式

    print(f"第{iter}代全局最短路径长度为：", historyfitness_best)
    iter += 1  # 迭代次数指示器+1

print(f"搜索到的全局最短路径长度为：", historyfitness_best)
print(f"搜索到的全局最短路径为：", historypath_best)

# 绘图
figure1 = plt.figure()  # 创建图形窗口 1
plot_tsp_path(coordinates, historypath_best)  # 路径结构图

figure2 = plt.figure()  # 创建图形窗口 2
plt.plot(historyfitness_table)
plt.xlabel('迭代次数')
plt.ylabel('路径距离')
plt.show()

