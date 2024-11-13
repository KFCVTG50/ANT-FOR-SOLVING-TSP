import copy
import random
import math
import numpy as np
import matplotlib.pyplot as plt


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

# 初始化种群并去重
def initpopulation(pop_size, city_number):
    popuation = []
    while len(popuation) < pop_size:
        temp = random.sample(range(city_number), city_number)
        if temp not in popuation:
            popuation.append(temp)
    return popuation

# 计算路径长度
def fitness(population, dis_mat):
    fitness = []
    for i in range(len(population)):
        distance = 0.0
        for j in range(city_number - 1):
            distance += dis_mat[population[i][j]][population[i][j + 1]]
        distance += dis_mat[population[i][-1]][population[i][0]]
        if distance == 0:
            f = float('inf')  # 如果路径长度为 0，则设置适应度为一个较大的数
        else:
            f = 1 / distance ** 2  # 计算路径长度平方的倒数作为适应度
        fitness.append(f)

    return fitness


# 选择函数，轮盘赌选择
def select(population, fitness):
    index = random.randint(0, pop_size - 1)
    s = 0
    r = random.uniform(0, sum(fitness))
    for i in range(len(population)):
        s += fitness[i]
        if s >= r:
            index = i
            break
    # print(population[index])
    return population[index]

# 交叉
def crossover(parent1, parent2):  # 传入两个父代
    if random.random() < pc:  # 按照一定的概率进行交叉操作
        chrom1 = parent1[:]  # 复制父代染色体
        chrom2 = parent2[:]
        # 交叉点，选择两个随机的交叉点。如果第一个点在第二个点的右侧，则交换两个点
        cpoint1 = random.randint(0, city_number - 1)
        cpoint2 = random.randint(0, city_number - 1)
        if cpoint1 > cpoint2:
            temp = cpoint1
            cpoint1 = cpoint2
            cpoint2 = temp

        # 未进行杂交之前，先保存两个父代个体的杂交段以及杂交段后面的片段
        # 保存cpoint1以及后面的片段
        temp1 = []
        temp2 = []
        for i in range(cpoint1, len(chrom1)):
            temp1.append(chrom1[i])
            temp2.append(chrom2[i])

        # 交叉操作，在交叉点之间对染色体进行交叉操作。
        for i in range(cpoint1, cpoint2 + 1):
            chrom1[i] = parent2[i]
            chrom2[i] = parent1[i]

        # 在杂交之后，先只保留每个父体杂交段以及杂交段以前的片段，然后在加上未杂交之前准备的杂交段以及杂交段后面的片段
        # 保存cpoint2以及前面的片段
        new_chrom1 = []
        new_chrom2 = []
        for i in range(cpoint2 + 1):
            new_chrom1.append(chrom1[i])
            new_chrom2.append(chrom2[i])
        new_chrom1.extend(temp1)
        new_chrom2.extend(temp2)

        # 现在通过映射的原理，去掉重复的城市点
        temporary1 = []
        temporary2 = []
        for i in range(len(new_chrom1)):
            if new_chrom1[i] not in temporary1:
                temporary1.append(new_chrom1[i])
        for i in range(len(new_chrom2)):
            if new_chrom2[i] not in temporary2:
                temporary2.append(new_chrom2[i])
        chrom1 = temporary1
        chrom2 = temporary2
        return chrom1, chrom2
    else:
        return parent1[:], parent2[:]  # 输出两个进行杂交操作后的两个个体

# 变异
def mutate(chrom):  # 变异函数
    if random.random() < pm:  # 按照一定的概率进行变异操作
        mpoint1 = random.randint(0, city_number - 1)  # 随机产生两个变异位置
        mpoint2 = random.randint(0, city_number - 1)
        # 交换变异点的基因位
        temp = chrom[mpoint1]
        chrom[mpoint1] = chrom[mpoint2]
        chrom[mpoint2] = temp
    return chrom

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

# 画出每代最好适应值的图像
def best_show(lx, ly, fit_history):
    a = []
    for i in range(len(fit_history)):
        a.append(math.sqrt(1 / fit_history[i]))
    plt.plot(range(len(a)), a)
    plt.xlabel("Generation")
    plt.ylabel("Best_path_size")
    plt.show()


# --------------------------------------------------------------------------------------------------
# 参数设置
N = 1000  # 终止条件：最大迭代次数
# 遗传算法参数
pop_size = 500  # 种群数
pc = 0.7  # 交叉概率
pm = 0.02  # 突变概率
# 读取数据
city_position = read_tsp_file('eil51.tsp')  # 添加城市坐标数据
city_number = len(city_position)  # 城市数

dis_mat = DistMat(city_number, city_position)  # 距离矩阵表

# 主程序
best_path = []
best_fitness = 0.0
Best_Fitness = []  # 用来存放每代种群中最个体的适应值
population = initpopulation(pop_size, city_number)  # 初始种群
fit_array = fitness(population, dis_mat)

for iter in range(N):
    iter += 1
    fit_array = fitness(population, dis_mat)  # 适应值列表
    max_fitness = max(fit_array)
    max_index = fit_array.index(max_fitness)
    lx = []
    ly = []

    for i in population[max_index][:]:
        i = int(i)
        lx.append(city_position[i][0])
        ly.append(city_position[i][1])

    if max_fitness > best_fitness:
        best_fitness = max_fitness
        best_path = population[max_index][:]
        x = copy.copy(lx)
        y = copy.copy(ly)

    Best_Fitness.append(best_fitness)

    new_population = []
    n = 0
    while n < pop_size:
        p1 = select(population, fit_array)
        p2 = select(population, fit_array)
        while p2 == p1:
            p2 = select(population, fit_array)
        # 交叉
        chrom1, chrom2 = crossover(p1, p2)
        # 变异
        chrom1 = mutate(chrom1)
        chrom2 = mutate(chrom2)
        new_population.append(chrom1)
        new_population.append(chrom2)
        n += 2
    population = new_population

    last_best_fitness = 0
    if last_best_fitness < math.sqrt(1 / best_fitness):
        last_best_fitness = math.sqrt(1 / best_fitness)
        last_best_path = best_path

    print(f"第{iter}代全局最短路径长度为：", last_best_fitness)
    # print("-------------------------------------------------" * 2)

print("最终全局最优路径为：", best_path)
print("最终全局最短路径长度为：", math.sqrt(1 / best_fitness))

# 画图
figure1 = plt.figure()  # 创建图形窗口 1
plot_tsp_path(city_position, best_path)

figure2 = plt.figure()  # 创建图形窗口 2
x.append(x[0])
y.append(y[0])
best_show(x, y, Best_Fitness)