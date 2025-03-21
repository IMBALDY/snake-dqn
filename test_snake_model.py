import pygame
import numpy as np
import torch
import time
import argparse
from snake_dqn import SnakeGame, DQNAgent
import os
import matplotlib.pyplot as plt
from datetime import datetime

# 游戏参数
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
GRID_SIZE = 20
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE
FPS = 10  # 降低FPS使游戏可视化变慢，便于观察

# 颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)
YELLOW = (255, 255, 0)

def parse_args():
    parser = argparse.ArgumentParser(description='测试贪吃蛇DQN模型')
    parser.add_argument('--model', type=str, default='best_snake_model.pth', help='模型路径')
    parser.add_argument('--episodes', type=int, default=5, help='测试回合数')
    parser.add_argument('--speed', type=int, default=10, help='游戏速度(FPS)')
    parser.add_argument('--save_stats', action='store_true', help='是否保存统计信息')
    return parser.parse_args()

def draw_grid(screen):
    # 绘制网格
    for x in range(0, SCREEN_WIDTH, GRID_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, SCREEN_HEIGHT), 1)
    for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (SCREEN_WIDTH, y), 1)

def save_statistics(scores, steps, rewards, avg_score, max_score):
    # 创建统计目录
    stats_dir = "stats"
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 绘制并保存图表
    plt.figure(figsize=(15, 10))
    
    # 分数图表
    plt.subplot(2, 2, 1)
    plt.plot(scores, 'g-')
    plt.title('Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    
    # 步数图表
    plt.subplot(2, 2, 2)
    plt.plot(steps, 'b-')
    plt.title('Steps')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    
    # 奖励图表
    plt.subplot(2, 2, 3)
    plt.plot(rewards, 'r-')
    plt.title('Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # 统计信息
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.text(0.1, 0.9, f'Average Score: {avg_score:.2f}', fontsize=12)
    plt.text(0.1, 0.8, f'Max Score: {max_score}', fontsize=12)
    plt.text(0.1, 0.7, f'Average Steps: {np.mean(steps):.2f}', fontsize=12)
    plt.text(0.1, 0.6, f'Average Reward: {np.mean(rewards):.2f}', fontsize=12)
    plt.text(0.1, 0.5, f'Total Episodes: {len(scores)}', fontsize=12)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(f"{stats_dir}/snake_stats_{timestamp}.png")
    print(f"Statistics chart saved to {stats_dir}/snake_stats_{timestamp}.png")

def main():
    args = parse_args()
    
    # 初始化pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Snake DQN Test')
    clock = pygame.time.Clock()
    
    # 创建游戏环境
    env = SnakeGame()
    
    # 创建智能体
    state_size = len(env.get_state())
    action_size = 3  # 直行、右转、左转
    agent = DQNAgent(state_size, action_size)
    
    # 加载模型
    try:
        agent.load(args.model)
        print(f"Successfully loaded model: {args.model}")
    except Exception as e:
        print(f"Failed to load model: {args.model}, Error: {str(e)}")
        print("Please train the model first or check the model path")
        pygame.quit()
        return
    
    # 统计信息
    all_scores = []
    all_steps = []
    all_rewards = []
    
    # 测试循环
    for episode in range(args.episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        done = False
        
        print(f"Episode {episode+1}/{args.episodes}")
        
        while not done:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # 如果要保存统计信息且已收集数据
                    if args.save_stats and all_scores:
                        save_statistics(
                            all_scores, 
                            all_steps, 
                            all_rewards, 
                            np.mean(all_scores), 
                            max(all_scores) if all_scores else 0
                        )
                    pygame.quit()
                    return
            
            # 选择动作（无探索）
            action = agent.select_action(state, training=False)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # 更新状态
            state = next_state
            total_reward += reward
            step += 1
            
            # 渲染
            screen.fill(BLACK)
            draw_grid(screen)  # 绘制网格
            
            # 绘制食物
            pygame.draw.rect(screen, RED, (env.food[0] * GRID_SIZE, env.food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            
            # 绘制蛇
            for i, (x, y) in enumerate(env.snake):
                color = GREEN if i == 0 else BLUE
                pygame.draw.rect(screen, color, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
                
                # 为蛇头添加眼睛效果
                if i == 0:
                    eye_size = GRID_SIZE // 5
                    # 根据方向绘制眼睛
                    if env.direction == 0:  # UP
                        pygame.draw.circle(screen, WHITE, (x * GRID_SIZE + GRID_SIZE // 3, y * GRID_SIZE + GRID_SIZE // 3), eye_size)
                        pygame.draw.circle(screen, WHITE, (x * GRID_SIZE + 2 * GRID_SIZE // 3, y * GRID_SIZE + GRID_SIZE // 3), eye_size)
                    elif env.direction == 1:  # RIGHT
                        pygame.draw.circle(screen, WHITE, (x * GRID_SIZE + 2 * GRID_SIZE // 3, y * GRID_SIZE + GRID_SIZE // 3), eye_size)
                        pygame.draw.circle(screen, WHITE, (x * GRID_SIZE + 2 * GRID_SIZE // 3, y * GRID_SIZE + 2 * GRID_SIZE // 3), eye_size)
                    elif env.direction == 2:  # DOWN
                        pygame.draw.circle(screen, WHITE, (x * GRID_SIZE + GRID_SIZE // 3, y * GRID_SIZE + 2 * GRID_SIZE // 3), eye_size)
                        pygame.draw.circle(screen, WHITE, (x * GRID_SIZE + 2 * GRID_SIZE // 3, y * GRID_SIZE + 2 * GRID_SIZE // 3), eye_size)
                    elif env.direction == 3:  # LEFT
                        pygame.draw.circle(screen, WHITE, (x * GRID_SIZE + GRID_SIZE // 3, y * GRID_SIZE + GRID_SIZE // 3), eye_size)
                        pygame.draw.circle(screen, WHITE, (x * GRID_SIZE + GRID_SIZE // 3, y * GRID_SIZE + 2 * GRID_SIZE // 3), eye_size)
            
            # 显示当前状态信息
            font = pygame.font.SysFont("arial", 15)
            info_text = font.render(f"Episode: {episode+1}/{args.episodes} Steps: {step} Score: {env.score}", True, WHITE)
            reward_text = font.render(f"Total Reward: {total_reward:.1f} Current Reward: {reward:.1f}", True, YELLOW)
            screen.blit(info_text, (10, 10))
            screen.blit(reward_text, (10, 30))
            
            pygame.display.flip()
            clock.tick(args.speed)
            
        # 收集统计数据
        all_scores.append(env.score)
        all_steps.append(step)
        all_rewards.append(total_reward)
        
        print(f"Episode {episode+1} completed, Score: {env.score}, Total Reward: {total_reward:.2f}, Steps: {step}")
        time.sleep(1)  # 回合结束暂停一秒
    
    # 显示最终统计信息
    avg_score = np.mean(all_scores)
    max_score = max(all_scores) if all_scores else 0
    print(f"\nTest Completed: {args.episodes} episodes")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Max Score: {max_score}")
    print(f"Average Steps: {np.mean(all_steps):.2f}")
    print(f"Average Reward: {np.mean(all_rewards):.2f}")
    
    # 保存统计信息
    if args.save_stats and all_scores:
        save_statistics(all_scores, all_steps, all_rewards, avg_score, max_score)
    
    # 显示结束消息
    font = pygame.font.SysFont("arial", 30)
    end_text = font.render("Test Completed, Press Any Key to Exit", True, WHITE)
    screen.blit(end_text, (SCREEN_WIDTH // 2 - end_text.get_width() // 2, SCREEN_HEIGHT // 2 - end_text.get_height() // 2))
    pygame.display.flip()
    
    # 等待用户按键退出
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                waiting = False
    
    pygame.quit()

if __name__ == '__main__':
    main()