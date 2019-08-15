import numpy as np
import gym
from DQN_reg import cartpole
from DQN_reg import Agent_e
from DQN_reg import Agent_r
from DQN_reg import Agent_f
from DQN_reg import Agent_a
import math

import os
import matplotlib.pyplot as plt

# Every model is experimented 5 times
iterations_Agent = 5
iterations = 10**4
test_trials = 50

# Test agentg
env = gym.make('CartPole-v1')
Agent_f_t = cartpole.make_model()
AF=[]
Agent_a_t = cartpole.make_model()
AA=[]
Agent_e_t = cartpole.make_model()
AE=[]
Agent_r_t = cartpole.make_model()
AR=[]
for i in range(5):
    Agent_f_t.load_weights(os.path.join('Agents_0.06', 'Agent_f_{}'.format(i+1)))
    AF.append(Agent_f_t)
    Agent_a_t.load_weights(os.path.join('Agents_0.06', 'Agent_a_{}'.format(i+1)))
    AA.append(Agent_a_t)
    Agent_e_t.load_weights(os.path.join('Agents_0.06', 'Agent_e_{}'.format(i+1)))
    AE.append(Agent_e_t)
    Agent_r_t.load_weights(os.path.join('Agents_0.06', 'Agent_r_{}'.format(i+1)))
    AR.append(Agent_r_t)

agents = [AF, AA, AE, AR]
for i in range(len(agents)):
    globals()['SCORE_{}'.format(i + 1)] = []
    all_score = list()
    for l in range(iterations_Agent):

        for j in range(5):
            state = env.reset()
            steps = 0
            score = 0
            globals()['scores_{}'.format(j + 1)] = []

            for k in range(iterations):
                action = cartpole.choose_best_action(agents[i][j],state)
                next_state, reward, is_terminal, _ = env.step(action)
                steps += 1
                if is_terminal:
                    reward = -100
                else:
                    reward = 2.4 - abs(next_state[0]) + 12 * 2 * math.pi / 360 - abs(next_state[2]) - abs(
                        next_state[1]) - abs(
                        next_state[3])

                next_state = np.array([next_state])[0, :]  # Process state so that it's a numpy array, shape (4,)
                score += reward
                state = next_state

                # If DONE, reset model, modify reward, record score
                if is_terminal:
                    env.reset()
                    globals()['scores_{}'.format(j + 1)].append(score)  # Record score
                    score = 0  # Reset score to zero
                elif steps >= 200:
                    globals()['scores_{}'.format(j + 1)].append(score)
                    score = 0
                    steps = 0
                    env.reset()

                if len(globals()['scores_{}'.format(j + 1)]) == test_trials:
                    break

            all_score.append(globals()['scores_{}'.format(j + 1)])
    # Take the mean score of all models & iterations
    for y in range(test_trials):
        globals()['SCORE_{}'.format(i + 1)].append((all_score[0][y] + all_score[1][y] + all_score[2][y] +
                                                    all_score[3][y] + all_score[4][y]+ all_score[5][y]+ all_score[6][y]+ all_score[7][y]+ all_score[8][y]+ all_score[9][y]) / (iterations_Agent*2))


fig = plt.figure('Testing Agents_new_0.2')
plt.plot(SCORE_1, label='Agent_f, mean score = {}'.format(round(np.mean(SCORE_1), 2)))
plt.plot(SCORE_2, label='Agent_e, mean score = {}'.format(round(np.mean(SCORE_2), 2)))
plt.plot(SCORE_3, label='Agent_e, mean score = {}'.format(round(np.mean(SCORE_3), 2)))
plt.plot(SCORE_4, label='Agent_r, mean score = {}'.format(round(np.mean(SCORE_4), 2)))
plt.legend()
plt.ylabel('Mean Scores')
plt.xlabel('Number of trials')
plt.title('Testing Results (mean score), ratio = {}'.format(round(ratio,3)))
plt.savefig('Test Agents_new_0.2, ratio = {}'.format(round(ratio,2)), format='png')
