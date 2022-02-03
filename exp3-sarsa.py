import numpy as np
import gym
import random
from collections import defaultdict
import matplotlib.pyplot as plt

env = gym.make("LunarLander-v2")
env.seed(0)

action_size = env.action_space.n
print("Action Size: ", action_size)

state_size = env.observation_space.shape
print("State Size: ", state_size[0])

class Sarsa_Agent():

    def __init__(self, gamma=0.95, learning_rate=0.8, epsilon=0.2, nepisodes=40000):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.nepisodes = nepisodes
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def greedy_policy(self, state):
        return np.argmax(self.Q[state])

    def epsilon_greedy_policy(self, state):
        action = 0
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(env.action_space.n)
        else:
            action = self.greedy_policy(state)
        return action

    def onpolicy_control(self):
        for episode in range(self.nepisodes):
            state = env.reset()
            state = state[0]
            done = False
            action = self.epsilon_greedy_policy(state)
            while not done:
                next_state, reward, done, info = env.step(action)
                next_action = self.epsilon_greedy_policy(next_state)
                self.Q[state][action] = self.Q[state][action] + self.learning_rate * (
                            reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])
                state = next_state
                action = next_action
            res = self.test_policy(200)
            if episode % 100 == 0:
                print(f'Episode: {episode} Success%: {res}')
            if res > 70:
                print(f'Solved! Episode: {episode} Success%: {res}')
                return self.Q
        return self.Q

    def test_policy(self, n):
        success = 0
        for episode in range(n):
            state = env.reset()
            done = False
            while not done:
                action = self.greedy_policy(state)
                state, reward, done, info = env.step(action)
            if reward == 1:
                success += 1
        return success / n * 100

a = Sarsa_Agent()
Q = a.onpolicy_control()
a.test_policy(100)