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

# 蚁群算法
def ant_colony_optimization(coordinates, numant, alpha, beta, rho, Q, itermax):
    numcity = len(coordinates)  # 城市个数
    etatable = 1.0 / (getdistmat(coordinates) + np.diag([1e10] * numcity))  # 启发函数矩阵
    pheromonetable = np.ones((numcity, numcity))  # 信息素矩阵
    pathtable = np.zeros((numant, numcity)).astype(int)  # 路径记录表
    distmat = getdistmat(coordinates)  # 城市的距离矩阵
    historyfitness_best = float('inf')
    historypath_best = []
    historyfitness_table = []

    for iter in range(itermax):
        random_integers = [random.randint(0, numcity-1) for _ in range(numant)]
        for i in range(pathtable.shape[0]):
            pathtable[i, 0] = random_integers[i]
        length = np.zeros(numant)  # 计算各个蚂蚁的路径距离

        for i in range(numant):
            visiting = pathtable[i, 0]  # 当前所在的城市
            unvisited = set(range(numcity))  # 未访问的城市
            unvisited.remove(visiting)  # 删除已访问的城市

            for j in range(1, numcity):  # 循环numcity-1次,访问剩余的numcity-1个城市
                # 每次用轮盘法选择下一个要访问的城市
                listunvisited = list(unvisited)
                probtrans = np.zeros(len(listunvisited))
                for k in range(len(listunvisited)):
                    probtrans[k] = np.power(pheromonetable[visiting][listunvisited[k]], alpha) \
                        * np.power(etatable[visiting][listunvisited[k]], beta)
                
                if np.sum(probtrans) == 0:
                    # 如果所有的转移概率都为零,则随机选择一个未访问的城市
                    k = random.choice(listunvisited)
                else:
                    cumsumprobtrans = (probtrans / np.sum(probtrans)).cumsum()
                    cumsumprobtrans -= np.random.rand()
                    k = listunvisited[np.where(cumsumprobtrans > 0)[0][0]]
                
                pathtable[i, j] = k  # 添加到路径表中
                unvisited.remove(k)  # 从未访问城市集合中删除已访问的城市
                length[i] += distmat[visiting][k]
                visiting = k

            length[i] += distmat[visiting][pathtable[i, 0]]  # 蚂蚁的路径距离包括最后一个城市和第一个城市的距离

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
                changepheromonetable[pathtable[i, j]][pathtable[i, j + 1]] += Q / distmat[pathtable[i, j]][pathtable[i, j + 1]]  # 计算信息素增量
            changepheromonetable[pathtable[i, j + 1]][pathtable[i, 0]] += Q / distmat[pathtable[i, j + 1]][pathtable[i, 0]]
        pheromonetable = (1 - rho) * pheromonetable + changepheromonetable  # 计算信息素公式

    return historyfitness_best, historyfitness_table

# 主程序
coordinates = read_tsp_file('eil51.tsp')  # 添加城市坐标数据
numant = 30  # 蚂蚁个数
alpha = 1  # 信息素重要程度因子
beta = 5  # 启发函数重要程度因子
Q = 1  # 信息素释放总量
itermax = 1000  # 循环最大值

rho_values = [0.2, 0.4, 0.6, 0.8]  # 不同的 ρ 值

best_fitness_values = []
fitness_tables = []

for rho in rho_values:
    print(f"当前 ρ 值为: {rho}")
    best_fitness, fitness_table = ant_colony_optimization(coordinates, numant, alpha, beta, rho, Q, itermax)
    best_fitness_values.append(best_fitness)
    fitness_tables.append(fitness_table)

# 绘图
figure1 = plt.figure()  # 创建图形窗口 1
for i, rho in enumerate(rho_values):
    plt.plot(fitness_tables[i], label=f"ρ = {rho}")
plt.xlabel('迭代次数')
plt.ylabel('路径距离')
plt.legend()

figure2 = plt.figure()  # 创建图形窗口 2
plt.bar(range(len(rho_values)), best_fitness_values)
plt.xlabel('ρ 值')
plt.ylabel('最佳路径距离')
plt.xticks(range(len(rho_values)), [f"{rho}" for rho in rho_values])

plt.show()