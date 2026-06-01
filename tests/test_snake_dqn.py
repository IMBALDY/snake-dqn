import os
import sys
import unittest

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from snake_dqn import DQNAgent, LEFT, SnakeGame


class SnakeDQNTests(unittest.TestCase):
    def test_state_has_expected_shape(self):
        env = SnakeGame()
        state = env.get_state()

        self.assertEqual(len(state), 20)
        self.assertEqual(state.dtype, np.float32)

    def test_food_does_not_spawn_on_snake(self):
        env = SnakeGame()

        for _ in range(100):
            env.generate_food()
            self.assertNotIn(env.food, env.snake)

    def test_wall_collision_ends_episode(self):
        env = SnakeGame()
        env.snake = [(0, 0)]
        env.direction = LEFT
        env.food = (10, 10)
        env.last_distance = env.get_distance_to_food()

        _, reward, done = env.step(0)

        self.assertTrue(done)
        self.assertEqual(reward, -50)

    def test_bundled_checkpoint_loads_when_present(self):
        model_path = os.path.join(ROOT_DIR, 'best_snake_model.pth')
        if not os.path.exists(model_path):
            self.skipTest('best_snake_model.pth is not present')

        env = SnakeGame()
        agent = DQNAgent(len(env.get_state()), action_size=3)
        agent.load(model_path)

        action = agent.select_action(env.get_state(), training=False)
        self.assertIn(action, (0, 1, 2))


if __name__ == '__main__':
    unittest.main()
