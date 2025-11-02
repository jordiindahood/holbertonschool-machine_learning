#!/usr/bin/env python3
"""Play an episode using a trained Q-table."""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays an episode using the trained Q-table.
    """
    state, _ = env.reset()
    rendered_outputs = [env.render()]
    total_rewards = 0

    for _ in range(max_steps):
        # Always exploit (choose best action)
        action = np.argmax(Q[state])
        new_state, reward, done, truncated, _ = env.step(action)

        total_rewards += reward
        rendered_outputs.append(env.render())

        state = new_state
        if done or truncated:
            break

    return total_rewards, rendered_outputs
