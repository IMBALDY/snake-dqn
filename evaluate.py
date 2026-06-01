#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Headless evaluation for a trained Snake DQN model.
"""

import argparse
import random

import numpy as np
import torch

from snake_dqn import DQNAgent, SnakeGame


def parse_args():
    parser = argparse.ArgumentParser(description='无窗口评估贪吃蛇DQN模型')
    parser.add_argument('--model', type=str, default='best_snake_model.pth', help='模型路径')
    parser.add_argument('--episodes', type=int, default=100, help='评估回合数')
    parser.add_argument('--max_steps', type=int, default=2000, help='每回合最大步数')
    parser.add_argument('--seed', type=int, default=123, help='随机种子')
    return parser.parse_args()


def evaluate(model_path, episodes, max_steps):
    env = SnakeGame()
    state_size = len(env.get_state())
    agent = DQNAgent(state_size, action_size=3)
    agent.load(model_path)

    scores = []
    steps = []
    rewards = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done and step < max_steps:
            action = agent.select_action(state, training=False)
            state, reward, done = env.step(action)
            total_reward += reward
            step += 1

        scores.append(env.score)
        steps.append(step)
        rewards.append(total_reward)

    return {
        'episodes': episodes,
        'avg_score': float(np.mean(scores)),
        'max_score': int(max(scores) if scores else 0),
        'avg_steps': float(np.mean(steps)),
        'avg_reward': float(np.mean(rewards)),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    metrics = evaluate(args.model, args.episodes, args.max_steps)

    print(f"Model: {args.model}")
    print(f"Episodes: {metrics['episodes']}")
    print(f"Average score: {metrics['avg_score']:.2f}")
    print(f"Max score: {metrics['max_score']}")
    print(f"Average steps: {metrics['avg_steps']:.2f}")
    print(f"Average reward: {metrics['avg_reward']:.2f}")


if __name__ == '__main__':
    main()
