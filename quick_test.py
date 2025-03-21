#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Snake DQN Quick Test Script
"""

import os
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description='Quick test for Snake DQN model')
    parser.add_argument('--model', type=str, default='best_snake_model.pth', help='Model path')
    parser.add_argument('--episodes', type=int, default=3, help='Number of test episodes')
    parser.add_argument('--speed', type=int, default=8, help='Game speed(FPS)')
    parser.add_argument('--save_stats', action='store_true', help='Save statistics')
    args = parser.parse_args()
    
    print("="*50)
    print("Snake DQN Quick Test")
    print("="*50)
    
    # Check if the model file exists
    if not os.path.exists(args.model):
        print(f"\nError: Model file '{args.model}' not found")
        print("Please train a model first or specify the correct model path. For example:")
        print("python quick_test.py --model=best_snake_model.pth")
        print("\nAvailable model files:")
        model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        if model_files:
            for model in model_files:
                print(f" - {model}")
        else:
            print("No model files (.pth) found")
        return
    
    # Check if the test script exists
    if not os.path.exists("test_snake_model.py"):
        print("\nError: Test script 'test_snake_model.py' not found")
        return
    
    # Build the command
    cmd = [
        sys.executable, 
        "test_snake_model.py",
        f"--model={args.model}",
        f"--episodes={args.episodes}",
        f"--speed={args.speed}"
    ]
    
    if args.save_stats:
        cmd.append("--save_stats")
    
    # Print test information
    print(f"\nUsing model: {args.model}")
    print(f"Test episodes: {args.episodes}")
    print(f"Game speed: {args.speed} FPS")
    print(f"Save statistics: {'Yes' if args.save_stats else 'No'}")
    print("\nStarting test...")
    
    try:
        # Start the test
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nTest manually terminated.")
    except Exception as e:
        print(f"\nError during test: {str(e)}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main() 