#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Snake DQN Training Launcher
"""

import subprocess
import sys
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Snake DQN Reinforcement Learning Trainer')
    parser.add_argument('--render', action='store_true', help='启用训练过程渲染', default=True)
    parser.add_argument('--render_freq', type=int, default=1, help='渲染频率（每多少回合渲染一次）')
    parser.add_argument('--fps', type=int, default=2000, help='训练时的FPS（帧率），值越大训练越快')
    parser.add_argument('--episodes', type=int, default=10000, help='训练回合数')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()

    print("="*50)
    print("Snake DQN Reinforcement Learning Trainer")
    print("="*50)
    print("\n训练设置:")
    print(f"- 训练回合数: {args.episodes}")
    print(f"- 渲染训练过程: {'是' if args.render else '否'}")
    if args.render:
        print(f"- 渲染频率: 每{args.render_freq}回合")
    print(f"- 训练速度(FPS): {args.fps}")
    
    print("\n最佳模型将保存为 'best_snake_model.pth'")
    print("最终模型将保存为 'final_snake_model.pth'")
    print("\n注意: 训练可能需要较长时间，请耐心等待。")
    print("您可以随时按Ctrl+C终止训练，最佳模型仍会保存。")
    
    # 检查必要文件
    required_files = ["snake_dqn.py", "test_snake_model.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"\n错误: 缺少以下文件: {', '.join(missing_files)}")
        input("按任意键退出...")
        return
    
    # 检查必要的Python包
    try:
        import pygame
        import torch
        import numpy
    except ImportError as e:
        print(f"\n错误: 缺少必要的Python包: {str(e)}")
        print("请安装必要的包:")
        print("pip install pygame torch numpy")
        input("按任意键退出...")
        return
    
    print("\n准备开始训练...")
    input("按Enter键开始训练...")
    
    try:
        # 构建命令行参数
        cmd = [sys.executable, "snake_dqn.py"]
        if args.render:
            cmd.append("--render")
        cmd.extend([
            "--render_freq", str(args.render_freq),
            "--fps", str(args.fps),
            "--episodes", str(args.episodes)
        ])
        
        # 启动训练脚本
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n训练被手动终止。")
    except Exception as e:
        print(f"\n训练过程中出错: {str(e)}")
    
    print("\n训练过程已结束。")
    
    # 检查模型文件是否生成
    if os.path.exists("best_snake_model.pth"):
        print("最佳模型已保存为 'best_snake_model.pth'")
        
        # 询问用户是否要测试模型
        test_model = input("\n是否要测试训练好的模型? (y/n): ").strip().lower()
        if test_model == 'y':
            try:
                subprocess.run([sys.executable, "test_snake_model.py"], check=True)
            except Exception as e:
                print(f"测试过程中出错: {str(e)}")
    else:
        print("警告: 未找到保存的模型文件。训练可能未成功完成。")
    
    print("\n程序已结束。")

if __name__ == "__main__":
    main()