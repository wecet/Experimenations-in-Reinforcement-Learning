import gym
import numpy as np
import cma
import matplotlib.pyplot as plt
from datetime import datetime

env = gym.make('LunarLander-v2')
nclasses = env.action_space.n
ninputs = env.observation_space.shape[0]
wlen = ninputs * nclasses
nepisodes = 20


def policy(observation, w):
    action = np.argmax( observation @ w )
    return action


def runEpisodeBatch(w, view=False, nepisodes=10):
    weights = np.reshape(w, [ninputs,nclasses]) 
    totReward = 0
    for i_episode in range(nepisodes):
        done = False
        observation = env.reset()
        while (not done):
            if view: env.render()
            action = policy(observation, weights)
            observation, reward, done, info = env.step(action)
            totReward += reward      
    f = totReward/nepisodes
    return f


def optim(optimf, nweights):
    rewards = []
    start = datetime.now()
    solution = np.zeros(nweights)
    theta_mean = np.zeros(nweights)
    theta_std = np.ones(nweights)
    population = 200
    n_elite = int(.05 * population)
    solved = False
    iter = 0
    extra_std = 2.0
    extra_decay_time = 5
    while not solved:
        iter += 1
        extra_cov = max(1.0 - iter / extra_decay_time, 0) * extra_std**2
        thetas = np.random.multivariate_normal(mean=theta_mean, cov=np.diag(np.array(theta_std**2) + extra_cov), size=population)
        rewards = [runEpisodeBatch(weights) for weights in thetas]

        elite_idxs = np.array(rewards).argsort()[-n_elite:]
        elite_thetas = thetas[elite_idxs]

        theta_mean = elite_thetas.mean(axis=0)
        theta_std = elite_thetas.std(axis=0)

        mean_res = np.mean(rewards)
        rewards[iter] = mean_res
        print("Episode: ", iter, "Average Reward: ", mean_res)
        if mean_res >= 195:
            end = datetime.now()
            solution = np.random.multivariate_normal(mean=theta_mean, cov=np.diag(np.array(theta_std**2)), size=1)
            print("Solved in: ", end-start)
            print("Solved in Episode: ", iter, "with an Average Reward: ", mean_res)
            solved = True

    plt.plot(np.arange(iter), rewards[0:iter])
    plt.show()

    return solution


bestw = optim(runEpisodeBatch, wlen)
runEpisodeBatch(bestw, True,10)
env.close()

