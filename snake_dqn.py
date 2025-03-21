import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import math
import time
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import argparse  # 添加命令行参数解析

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Snake DQN Training')
    parser.add_argument('--render', action='store_true', help='启用训练过程渲染', default=True)
    parser.add_argument('--render_freq', type=int, default=1, help='渲染频率（每多少回合渲染一次）')
    parser.add_argument('--fps', type=int, default=2000, help='训练时的FPS（帧率）')
    parser.add_argument('--episodes', type=int, default=10000, help='训练回合数')
    return parser.parse_args()

# 设置随机种子
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

# 游戏参数
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
GRID_SIZE = 20
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE
FPS = 2000  # 增加FPS以加快训练速度

# 颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# DQN参数
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 10000
TARGET_UPDATE = 10
MEMORY_SIZE = 100000
LEARNING_RATE = 0.0001

# 蛇的方向
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Q网络模型
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 经验回放内存
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, next_state, reward, done = zip(*batch)
        return state, action, next_state, reward, done
    
    def __len__(self):
        return len(self.memory)

# 贪吃蛇游戏环境
class SnakeGame:
    def __init__(self):
        self.reset()
        
    def reset(self):
        # 初始化蛇的位置和方向
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = RIGHT
        self.score = 0
        self.steps = 0
        self.last_distance = 0
        self.last_positions = []  # 记录最近的位置，用于检测原地打转
        self.generate_food()
        return self.get_state()
    
    def generate_food(self):
        while True:
            self.food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if self.food not in self.snake:
                break
        # 计算初始距离
        self.last_distance = self.get_distance_to_food()
    
    def get_distance_to_food(self):
        head = self.snake[0]
        return math.sqrt((head[0] - self.food[0])**2 + (head[1] - self.food[1])**2)
    
    def get_state(self):
        # 构建状态表示
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # 检查四个方向是否有障碍物（墙或蛇身）
        danger_straight = False
        danger_right = False
        danger_left = False
        
        # 根据当前方向判断危险
        if self.direction == UP:
            danger_straight = head_y == 0 or (head_x, head_y - 1) in self.snake
            danger_right = head_x == GRID_WIDTH - 1 or (head_x + 1, head_y) in self.snake
            danger_left = head_x == 0 or (head_x - 1, head_y) in self.snake
        elif self.direction == RIGHT:
            danger_straight = head_x == GRID_WIDTH - 1 or (head_x + 1, head_y) in self.snake
            danger_right = head_y == GRID_HEIGHT - 1 or (head_x, head_y + 1) in self.snake
            danger_left = head_y == 0 or (head_x, head_y - 1) in self.snake
        elif self.direction == DOWN:
            danger_straight = head_y == GRID_HEIGHT - 1 or (head_x, head_y + 1) in self.snake
            danger_right = head_x == 0 or (head_x - 1, head_y) in self.snake
            danger_left = head_x == GRID_WIDTH - 1 or (head_x + 1, head_y) in self.snake
        elif self.direction == LEFT:
            danger_straight = head_x == 0 or (head_x - 1, head_y) in self.snake
            danger_right = head_y == 0 or (head_x, head_y - 1) in self.snake
            danger_left = head_y == GRID_HEIGHT - 1 or (head_x, head_y + 1) in self.snake
        
        # 食物相对位置（基于当前方向）
        food_left = food_right = food_up = food_down = False
        
        if food_x < head_x:
            food_left = True
        elif food_x > head_x:
            food_right = True
            
        if food_y < head_y:
            food_up = True
        elif food_y > head_y:
            food_down = True
        
        # 构建状态向量
        state = [
            # 危险
            danger_straight,
            danger_right,
            danger_left,
            
            # 当前方向
            self.direction == LEFT,
            self.direction == RIGHT,
            self.direction == UP,
            self.direction == DOWN,
            
            # 食物位置
            food_left,
            food_right,
            food_up,
            food_down,
            
            # 添加归一化的头部和食物坐标
            head_x / GRID_WIDTH,
            head_y / GRID_HEIGHT,
            food_x / GRID_WIDTH,
            food_y / GRID_HEIGHT,
            
            # 添加蛇的长度（归一化）
            len(self.snake) / (GRID_WIDTH * GRID_HEIGHT)
        ]
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        # 0: 直行, 1: 右转, 2: 左转
        # 更新方向
        if action == 1:  # 右转
            self.direction = (self.direction + 1) % 4
        elif action == 2:  # 左转
            self.direction = (self.direction - 1) % 4
        
        # 移动蛇头
        head_x, head_y = self.snake[0]
        if self.direction == UP:
            head_y -= 1
        elif self.direction == RIGHT:
            head_x += 1
        elif self.direction == DOWN:
            head_y += 1
        elif self.direction == LEFT:
            head_x -= 1
        
        # 记录新位置
        self.steps += 1
        head = (head_x, head_y)
        self.last_positions.append(head)
        if len(self.last_positions) > 10:  # 保留最近10步
            self.last_positions.pop(0)
        
        # 检查是否撞墙或自己
        done = False
        reward = 0
        if (head_x < 0 or head_x >= GRID_WIDTH or 
            head_y < 0 or head_y >= GRID_HEIGHT or 
            head in self.snake):
            reward = -50
            done = True
        else:
            # 更新蛇身
            self.snake.insert(0, head)
            
            # 检查是否吃到食物
            if head == self.food:
                self.score += 1
                reward = 10
                self.generate_food()
            else:
                self.snake.pop()
                
                # 计算靠近/远离食物的奖励
                current_distance = self.get_distance_to_food()
                if current_distance < self.last_distance:
                    reward += 0.1
                else:
                    reward -= 0.1
                self.last_distance = current_distance
                
                # 检测原地打转
                if len(self.last_positions) >= 8:
                    unique_positions = set(self.last_positions)
                    if len(unique_positions) <= 2:  # 如果最近的位置只有2个或更少的不同点
                        reward -= 1
        
        next_state = self.get_state()
        return next_state, reward, done
    
    def render(self, screen):
        screen.fill(BLACK)
        
        # 绘制食物
        pygame.draw.rect(screen, RED, (self.food[0] * GRID_SIZE, self.food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        
        # 绘制蛇
        for i, (x, y) in enumerate(self.snake):
            color = GREEN if i == 0 else BLUE
            pygame.draw.rect(screen, color, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        
        # 显示分数
        font = pygame.font.SysFont("arial", 20)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        pygame.display.flip()

# DQN智能体
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Q网络
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        
        # 经验回放
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        # 探索参数
        self.steps_done = 0
        
    def select_action(self, state, training=True):
        # epsilon-greedy策略
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-self.steps_done / EPS_DECAY)
        self.steps_done += 1
        
        if training and sample < eps_threshold:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        # 从经验回放中采样
        states, actions, next_states, rewards, dones = self.memory.sample(BATCH_SIZE)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + (GAMMA * next_q * (1 - dones))
        
        # 计算损失
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.target_net.eval()

# 主函数
def main():
    # 解析命令行参数
    args = parse_args()
    
    # 初始化pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Snake DQN')
    clock = pygame.time.Clock()
    
    # 设置FPS（可通过命令行参数修改）
    training_fps = args.fps
    
    # 创建游戏环境
    env = SnakeGame()
    
    # 创建智能体
    state_size = len(env.get_state())
    action_size = 3  # 直行、右转、左转
    agent = DQNAgent(state_size, action_size)
    
    # 训练参数
    num_episodes = args.episodes
    max_steps = 2000
    
    # 渲染设置
    render_training = args.render  # 通过命令行参数控制是否渲染
    render_frequency = args.render_freq  # 通过命令行参数控制渲染频率
    
    # 记录训练进度
    scores = []
    avg_scores = []
    best_score = 0
    
    # 创建统计图表目录
    stats_dir = "stats"
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    
    # 创建实时图表
    plt.ion()  # 开启交互模式
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    fig.tight_layout(pad=3.0)
    
    # 设置窗口标题（兼容不同平台）
    try:
        fig.canvas.set_window_title('Snake DQN Training Progress')
    except:
        try:
            fig.canvas.manager.set_window_title('Snake DQN Training Progress')
        except:
            pass  # 如果都失败，就不设置标题
    
    # 准备绘图数据
    episodes = []
    plot_scores = []
    plot_avg_scores = []
    plot_max = 0
    
    print("开始训练10000轮...")
    
    # 训练循环
    for episode in range(num_episodes):
        state = env.reset()
        score = 0
        done = False
        step = 0
        
        while not done and step < max_steps:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    plt.close()
                    return
            
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # 记录经验
            agent.memory.push(state, action, next_state, reward, done)
            
            # 学习
            loss = agent.learn()
            
            # 状态更新
            state = next_state
            score += reward
            step += 1
            
            # 渲染游戏画面（如果启用）
            if render_training and episode % render_frequency == 0:
                env.render(screen)
                clock.tick(training_fps)
        
        # 记录分数
        scores.append(score)
        episodes.append(episode)
        plot_scores.append(env.score)
        
        # 更新目标网络
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()
        
        # 计算平均分数（最近100轮）
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        avg_scores.append(avg_score)
        plot_avg_scores.append(avg_score)
        plot_max = max(plot_max, env.score)
        
        # 每100轮打印一次训练信息，减少打印频率
        if episode % 100 == 0 or episode == num_episodes - 1:
            print(f'Episode: {episode+1}/{num_episodes}, Score: {env.score}, Avg Score: {avg_score:.2f}, Steps: {step}')
        
        # 保存最佳模型
        if env.score > best_score:
            best_score = env.score
            agent.save('best_snake_model.pth')
            print(f"新的最佳分数: {best_score}，模型已保存")
        
        # 更新图表（每10轮更新一次，减少性能开销）
        if episode % 10 == 0:
            update_plot(ax1, ax2, episodes, plot_scores, plot_avg_scores, plot_max, episode, num_episodes, best_score)
            # 保存当前图表
            if episode % 1000 == 0 and episode > 0:
                plt.savefig(f"{stats_dir}/training_progress_{episode}.png")
    
    # 训练结束后保存最终模型和图表
    agent.save('final_snake_model.pth')
    plt.savefig(f"{stats_dir}/final_training_progress.png")
    print(f"训练完成！最终模型已保存到 'final_snake_model.pth'")
    print(f"最高分数: {best_score}")
    print(f"训练进度图表已保存到 {stats_dir}/final_training_progress.png")
    
    # 关闭图表的交互模式
    plt.ioff()
    plt.close()
    
    # 训练完成后，让用户选择是否要测试模型
    pygame.display.set_caption('Training Completed - Press Any Key to Continue')
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            elif event.type == pygame.KEYDOWN:
                waiting = False
                # 运行测试脚本
                print("开始测试模型...")
                try:
                    import test_snake_model
                    test_snake_model.main()
                except ImportError:
                    print("找不到测试脚本，请手动运行 'python test_snake_model.py'")
    
    pygame.quit()

# 更新实时图表
def update_plot(ax1, ax2, episodes, scores, avg_scores, plot_max, current_episode, total_episodes, best_score):
    ax1.clear()
    ax2.clear()
    
    # 设置第一个子图 - 分数
    ax1.set_title('Training Progress')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.plot(episodes, scores, 'b-', label='Score')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # 设置第二个子图 - 平均分数
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Score (Last 100 Episodes)')
    ax2.plot(episodes, avg_scores, 'r-')
    ax2.grid(True)
    
    # 添加进度文本
    progress_txt = f'Episode: {current_episode}/{total_episodes}\nBest Score: {best_score}'
    ax1.text(0.01, 0.99, progress_txt, transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.draw()
    plt.pause(0.001)  # 短暂暂停以更新图表

if __name__ == '__main__':
    main() 