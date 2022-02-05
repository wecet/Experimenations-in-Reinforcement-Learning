import matplotlib
import numpy as np
import gym
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
import sklearn.pipeline
import sklearn.preprocessing
import pickle

env = gym.make('LunarLander-v2')

num_episodes = 5000
discount_factor = 0.99
alpha = 0.1
nA = env.action_space.n

#Parameter vector define number of parameters per action based on featurizer size
w = np.zeros((nA,400))

# Plots
plt_actions = np.zeros(nA)
episode_rewards = np.zeros(num_episodes)

# Get satistics over observation space samples for normalization
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Create radial basis function sampler to convert states to features for nonlinear function approx
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
		])
# Fit featurizer to our scaled inputs
featurizer.fit(scaler.transform(observation_examples))
#featurizer.fit(observation_examples)

# Normalize and turn into feature

def featurize_state(state):
	# Transform data
	scaled = scaler.transform([state])
	featurized = featurizer.transform(scaled)
	#featurized = featurizer.transform([state])
	return featurized

def Q(state,action,w):
	value = state.dot(w[action])
	return value

# Epsilon greedy policy
def policy(state, weight, epsilon=0.001):
	A = np.ones(nA,dtype=float) * epsilon/nA
	best_action =  np.argmax([Q(state,a,w) for a in range(nA)])
	A[best_action] += (1.0-epsilon)
	sample = np.random.choice(nA,p=A)
	return sample

# Helper function save params
def save_params(fname, param_list):
    file = open(fname+'.obj', 'wb')
    pickle.dump(param_list, file)
    file.close()

# Helper function load params
def load_params(fname):
    file = open(fname+'.obj', 'rb')
    param_list = pickle.load(file)
    return param_list


# Our main training loop
mov_avg_result = 0.

for e in range(num_episodes):

    state = env.reset()
    state = featurize_state(state)

    while True:

        # env.render()
        # Sample from our policy
        action = policy(state, w)

        # Statistic for graphing
        plt_actions[action] += 1
        # Step environment and get next state and make it a feature
        next_state, reward, done, _ = env.step(action)
        next_state = featurize_state(next_state)

        # Figure out what our policy tells us to do for the next state
        next_action = policy(next_state, w)

        # Statistic for graphing
        episode_rewards[e] += reward

        # Figure out target and td error
        target = reward + discount_factor * Q(next_state, next_action, w)
        td_error = target - Q(state, action, w)

        # Find gradient with code to check it commented below (check passes)
        dw = (td_error).dot(state)

        # Update weight
        w[action] += alpha * dw

        if done:
            break
        # update our state
        state = next_state

    if e > 100:
        mov_avg_result = np.mean(episode_rewards[e - 100:e])
        if mov_avg_result >= 195:
            print(f'Solved! Episode: {e} Average Score: {mov_avg_result}')
            save_params('weights2', [w, scaler, featurizer])
            break

    if e > 0 and e % 100 == 0:
        print(f'Episode: {e} Average Score: {mov_avg_result}')


plt.plot(np.arange(e),episode_rewards[0:e])
plt.show()

env.close()

w, scaler, featurizer = load_params('weights2')

# test our trained model
for episodes in range(10):
	done = False
	s = env.reset()
	while not done:
		s = featurize_state(s)
		env.render()
		a = policy(s, w, 0)
		s, r, done, info = env.step(a)

env.close()