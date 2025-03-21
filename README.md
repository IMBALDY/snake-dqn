# 🐍 贪吃蛇 DQN 强化学习

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.6%2B-blue" alt="Python 3.6+">
  <img src="https://img.shields.io/badge/PyTorch-Latest-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</div>

<p align="center">
  <img src="./stats/snake_demo.gif" width="300" alt="贪吃蛇DQN演示" />
</p>

本项目使用深度Q网络（DQN）强化学习算法训练智能体自主玩贪吃蛇游戏。通过DQN算法，AI能够学习如何根据当前游戏状态做出最佳决策，最终达到高分游戏表现。

## 📋 特性

- 基于**PyTorch**实现DQN（深度Q网络）强化学习算法
- 使用**Pygame**进行游戏环境可视化
- 精心设计的奖励机制
  - 靠近食物：+0.1奖励
  - 远离食物：-0.1惩罚
  - 吃到食物：+10奖励
  - 撞墙/撞到自己：-50惩罚
  - 原地打转：-1惩罚（防止循环行为）
- 训练过程可视化
  - 实时图表显示训练指标
  - 自动保存阶段性训练进度
- 完整的模型保存与加载机制
- 详细的性能统计和评估工具

## 🔧 环境要求

- Python 3.6+
- PyTorch
- Pygame
- NumPy
- Matplotlib

## 🚀 快速开始

### 安装依赖

```bash
pip install pygame torch numpy matplotlib
```

### 克隆仓库

```bash
git clone https://github.com/imbaldy/snake-dqn.git
cd snake-dqn
```

### 开始训练

```bash
python start_training.py
```

### 测试训练好的模型

```bash
python test_snake_model.py
```

## 📖 详细使用说明

### 项目结构

```
snake-dqn/
├── snake_dqn.py        # 主程序文件 (游戏环境+DQN模型)
├── start_training.py   # 训练启动脚本
├── test_snake_model.py # 模型测试脚本
├── quick_test.py       # 快速测试脚本
├── stats/              # 训练统计与图表
└── README.md           # 项目文档
```

### 训练过程

训练会自动进行10000轮，但你可以随时中断。训练过程中最佳模型会自动保存。

#### 训练速度设置

训练时可以通过命令行参数控制训练速度：

```bash
# 快速训练模式（默认）
python start_training.py --fps 2000

# 观察模式（便于观察蛇的行为）
python start_training.py --fps 60
```

默认FPS值为2000，用于加速训练过程。如果想观察训练过程中蛇的行为，建议将FPS设置为60左右，可以清晰地看到蛇的移动情况。

#### 训练参数选项

- `--render`: 是否渲染游戏画面（默认：开启）
- `--render_freq`: 渲染频率（每多少回合渲染一次）
- `--fps`: 训练帧率（默认：2000，值越大训练越快）
- `--episodes`: 训练回合数（默认：10000）

示例：

```bash
# 不渲染画面的快速训练
python start_training.py --render False --fps 5000

# 低频渲染的训练（每50回合渲染一次）
python start_training.py --render --render_freq 50 --fps 2000
```

### 测试模型

#### 详细测试

使用`test_snake_model.py`进行详细测试：

```bash
python test_snake_model.py --model best_snake_model.pth --episodes 10 --speed 10 --save_stats
```

参数说明：
- `--model`: 模型文件路径（默认：best_snake_model.pth）
- `--episodes`: 测试回合数（默认：5）
- `--speed`: 游戏速度(FPS)（默认：10）
- `--save_stats`: 生成并保存性能统计图表

#### 快速测试

使用`quick_test.py`快速查看模型效果：

```bash
python quick_test.py --model best_snake_model.pth
```

## 💡 算法架构

### 网络架构

DQN网络使用三层全连接神经网络：
- 输入层：17个神经元（表示游戏状态）
- 隐藏层1：128个神经元
- 隐藏层2：128个神经元
- 输出层：3个神经元（表示动作：直行、左转、右转）

### 状态表示

游戏状态由以下特征组成：
- 危险检测（3个特征）：前方、右侧、左侧是否有障碍物
- 当前方向（4个特征）
- 食物相对位置（4个特征）
- 蛇头和食物的坐标（4个特征，归一化）
- 蛇的长度（1个特征，归一化）

### 训练参数

- 批次大小：64
- 折扣因子：0.99
- 初始探索率：1.0
- 最终探索率：0.01
- 探索率衰减步数：10000
- 学习率：0.0001
- 经验回放容量：100000
- 目标网络更新频率：10个回合

## 📊 性能评估

训练过程中会生成两种统计数据：

1. **实时训练进度图表**：展示每回合得分和平均得分曲线
2. **测试阶段统计图表**：包含详细的性能指标，如每回合分数、步数和总奖励

## 📄 许可证

本项目基于MIT许可证发布。详见[LICENSE](LICENSE)文件。

## 👥 贡献

欢迎提交Issue和Pull Request！如有任何问题或建议，请随时联系。

## 📬 联系方式

- GitHub: [imbaldy](https://github.com/imbaldy)
- Email: 873312124@qq.com

---

<p align="center">如果您喜欢这个项目，请给它一个⭐️！</p>
