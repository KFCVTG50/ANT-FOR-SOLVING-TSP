import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# 计算各城市之间的距离，得到距离矩阵
def DistMat(n, cities):  # n为城市个数， cities为城市坐标
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist[i][j] = dist[j][i] = np.linalg.norm(cities[i] - cities[j])  # linalg.norm 求向量的范数（默认求 二范数），得到 i、j 间的距离
    return dist

# 计算距离矩阵
def clac_distance(X, Y):
    """
    计算两个城市之间的欧氏距离，二范数
    :param X: 城市X的坐标.np.array数组
    :param Y: 城市Y的坐标.np.array数组
    :return:
    """
    distance_matrix = np.zeros((city_num, city_num))
    for i in range(city_num):
        for j in range(city_num):
            if i == j:
                continue

            distance = np.sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
            distance_matrix[i][j] = distance

    return distance_matrix


# 定义总距离(路程即适应度值)
def fitness_func(distance_matrix, x_i):
    """
    适应度函数
    :param distance_matrix: 城市距离矩阵
    :param x_i: PSO的一个解（路径序列）
    :return:
    """
    total_distance = 0
    for i in range(1, city_num):
        start_city = x_i[i - 1]
        end_city = x_i[i]
        total_distance += distance_matrix[start_city][end_city]
    total_distance += distance_matrix[x_i[-1]][x_i[0]]  # 从最后的城市返回出发的城市

    return total_distance


# 定义速度更新函数
def get_ss(x_best, x_i, r):
    """
    计算交换序列，即x2结果交换序列ss得到x1，对应PSO速度更新公式中的 r1(pbest-xi) 和 r2(gbest-xi)
    :param x_best: pbest or gbest
    :param x_i: 粒子当前的解
    :param r: 随机因子
    :return:
    """
    velocity_ss = []
    for i in range(len(x_i)):
        if x_i[i] != x_best[i]:
            j = np.where(x_i == x_best[i])[0][0]
            so = (i, j, r)  # 得到交换子
            velocity_ss.append(so)
            x_i[i], x_i[j] = x_i[j], x_i[i]  # 执行交换操作

    return velocity_ss


# 定义位置更新函数
def do_ss(x_i, ss):
    """
    执行交换操作
    :param x_i:
    :param ss: 由交换子组成的交换序列
    :return:
    """
    for i, j, r in ss:
        rand = np.random.random()
        if rand <= r:
            x_i[i], x_i[j] = x_i[j], x_i[i]
    return x_i

# 绘制路径图
def plot_tsp_path(cities, path):
    plt.plot(*zip(*cities), 'co')
    for i in range(len(path) - 1):
        x, y = zip(cities[path[i]], cities[path[i+1]])
        plt.plot(x, y, 'b')
    x, y = zip(cities[path[-1]], cities[path[0]])
    for i, city in enumerate(cities):  
        plt.annotate(str(i), (city[0], city[1]))
    plt.plot(x, y, 'b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    city_position = read_tsp_file('eil51.tsp') 
    city_num = len(city_position)  

    distance_matrix = DistMat(city_num, city_position)  

    size = 500  
    r1 = 0.7  
    iter_max_num = 1000  
    fitness_value_lst = []

    pbest_init = np.zeros((size, city_num), dtype=np.int64)
    for i in range(size):
        pbest_init[i] = np.random.choice(list(range(city_num)), size=city_num, replace=False)

    pbest = pbest_init
    pbest_fitness = np.zeros((size, 1))
    for i in range(size):
        pbest_fitness[i] = fitness_func(distance_matrix, x_i=pbest_init[i])

    gbest = pbest_init[pbest_fitness.argmin()]
    gbest_fitness = pbest_fitness.min()

    fitness_value_lst.append(gbest_fitness)

    for i in range(iter_max_num):
        for j in range(size):
            pbest_i = pbest[j].copy()
            x_i = pbest_init[j].copy()

            ss1 = get_ss(pbest_i, x_i, r1)
            ss2 = get_ss(gbest, x_i, r1)
            ss = ss1 + ss2
            x_i = do_ss(x_i, ss)

            fitness_new = fitness_func(distance_matrix, x_i)
            fitness_old = pbest_fitness[j]
            if fitness_new < fitness_old:
                pbest_fitness[j] = fitness_new
                pbest[j] = x_i

            gbest_fitness_new = pbest_fitness.min()
            gbest_new = pbest[pbest_fitness.argmin()]
            if gbest_fitness_new < gbest_fitness:
                gbest_fitness = gbest_fitness_new
                gbest = gbest_new
        fitness_value_lst.append(gbest_fitness)
        print("第%d迭代，最优值为：%s" % (i, gbest_fitness))

    # Output
    print("===最优路线：", gbest)
    print("===最优值：", gbest_fitness)

    # Plot
    figure1 = plt.figure() 
    plot_tsp_path(city_position, gbest)  

    figure2 = plt.figure()  
    plt.plot(fitness_value_lst)
    plt.xlabel('迭代次数')
    plt.ylabel('路径距离')
    plt.show()
