import numpy as np 
import gym   
import random
from collections import defaultdict

env = gym.make("Taxi-v3")
env.render()

action_size = env.action_space.n
print("Action Size: ", action_size)

state_size = env.observation_space.n
print("State Size: ", state_size)

qtable = np.zeros((state_size, action_size))
total_episodes = 50000        # Total episodes
total_test_episodes = 100     # Total test episodes
max_steps = 99                # Max steps per episode

learning_rate = 0.7           # Learning rate
gamma = 0.618                 # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.01             # Exponential decay rate for exploration prob

# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False

    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0,1)

        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])

        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma *
                                    np.max(qtable[new_state, :]) - qtable[state, action])

        # Our new state is state
        state = new_state

        # If done : finish episode
        if done == True:
            break

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    
env.reset()
rewards = []

for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        env.render()
        action = np.argmax(qtable[state,:])

        new_state, reward, done, info = env.step(action)

        total_rewards += reward

        if done:
            rewards.append(total_rewards)
            break
        state = new_state
env.close()
score = sum(rewards)/total_test_episodes
print("Score: " + str(score))
print("End of Q-Learning ------------------")
print("")
print("Start of SARSA ---------------------")


class SARSA():

    def __init__(self, gamma=0.95, learning_rate=0.9, epsilon=0.2, nepisodes=10000):
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

a = SARSA()
Q = a.onpolicy_control()
a.test_policy(100)