# Experimenations-in-Reinforcement-Learning

## Experiment 1 - Bandit Algorithms

For the first experiment, the objective was to investigate
whether adding context to the problem improved the long term rewards. To validate this experiment multiple bandit
algorithms (discussed in Chapter 2.3) being LinUCB, non contextual UCB, Thompson Sampling and epsilon-greedy
methods were compared to the LinUCB implementation.
The linear UCB contextual bandit, where the payout is
believed to be a linear function of the context features, was
created using the UCB algorithm, which is popular in the
MAB (With a MAB approach, the variants are treated similarly) environment. A simulation experiment with varied
alpha values of a LinUCB policy was also conducted.
All of the algorithms surrounding experiment 1 were all
carried out in the context of the dataset provided, in which
the arms and the rewards were segmented into separate lists
in pre-processing.

## Experiment 2 - Taxi v3

The second experiment tries to solve the OpenAI Gym Taxi-v3 environment. The environment can be considered solved with an
average reward between 9 and 9.5. In the tackling of this environment, the SARSA and Q-Learning algorithms were considered in the solution.
![image](https://user-images.githubusercontent.com/73174341/154957852-49457950-62ab-45b7-bc0b-468b8e762498.png)


## Experiment 3 - LunarLander v2

The third experiment tries to solve the OpenAI Gym LunarLander-v2 environment. The environment can be considered solved with an
average reward above 195. In the tackling of this environment, the SARSA and Q-Learning algorithms alongside the more traditional CEM were considered in the solution.
![image](https://user-images.githubusercontent.com/73174341/154957793-4043fe73-4366-487d-a449-1285854663a5.png)
