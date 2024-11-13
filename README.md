# TSP Solving Algorithms

本项目包含多种用于解决旅行商问题（TSP）的算法实现，包括蚁群算法（ACO）、遗传算法（GA）、模拟退火算法（SA）、粒子群优化算法（PSO）等。

## 文件结构

- `ANT/ACO.py`: 蚁群算法的实现，用于解决TSP。
- `ANT/ss.py`: 动态调整信息素挥发率的蚁群算法实现，适用于动态TSP（DTSP）。
- `GA/GA.py`: 遗传算法的实现，用于解决TSP。
- `SA/SA.py`: 模拟退火算法的实现，用于解决TSP。
- `PSO/PSO.py`: 粒子群优化算法的实现，用于解决TSP。

## 依赖

运行这些代码需要以下Python库：

- `numpy`
- `matplotlib`
- `random`
- `math`
- `pandas`（仅用于PSO）

可以通过以下命令安装这些库：
```bash
pip install numpy matplotlib pandas
```

## 使用方法

### 运行蚁群算法

1. 确保 `ANT/ACO.py` 和 `ANT/eil76.tsp` 文件在同一目录下。
2. 在终端中运行：

   ```bash
   python ANT/ACO.py
   ```

### 运行动态蚁群算法（适用于DTSP）

1. 确保 `ANT/ss.py` 和 `ANT/eil76.tsp` 文件在同一目录下。
2. 在终端中运行：

   ```bash
   python ANT/ss.py
   ```

   **改进说明**: `ss.py` 文件中实现了动态调整信息素挥发率（ρ）的功能，以适应动态TSP（DTSP）。在每次迭代中，算法会检测城市坐标的变化，并根据变化情况调整信息素挥发率，从而提高算法在动态环境下的适应性。

### 运行遗传算法

1. 确保 `GA/GA.py` 和 `eil51.tsp` 文件在同一目录下。
2. 在终端中运行：

   ```bash
   python GA/GA.py
   ```

### 运行模拟退火算法

1. 确保 `SA/SA.py` 和 `eil51.tsp` 文件在同一目录下。
2. 在终端中运行：

   ```bash
   python SA/SA.py
   ```

### 运行粒子群优化算法

1. 确保 `PSO/PSO.py` 和 `eil51.tsp` 文件在同一目录下。
2. 在终端中运行：

   ```bash
   python PSO/PSO.py
   ```

## 数据文件

- `eil51.tsp` 和 `eil76.tsp` 是标准的TSP数据文件，包含城市的坐标信息。

