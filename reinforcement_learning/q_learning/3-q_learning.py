#!/usr/bin/env python3
"""
Implements the Q-learning algorithm for FrozenLake
"""

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100,
          alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning.
    """
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        for _ in range(max_steps):
            # Choose action (explore or exploit)
            action = epsilon_greedy(Q, state, epsilon)

            # Take action
            new_state, reward, done, truncated, _ = env.step(action)

            # Custom reward: -1 if agent falls in a hole (done but not goal)
            if done and reward == 0.0:
                reward = -1

            # Q-learning update rule
            best_next_action = np.argmax(Q[new_state])
            td_target = reward + gamma * Q[new_state, best_next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = new_state
            episode_reward += reward

            if done or truncated:
                break

        # Decay epsilon (but not below min_epsilon)
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay))

        total_rewards.append(episode_reward)

    return Q, total_rewards
