from DQN_reg import RNN
from DQN_reg import cartpole
import argparse
import os
import random
import logging

import gym
import numpy as np
import keras
import matplotlib.pyplot as plt
import math


STATE_SHAPE = (4,) # This is the shape after pre-processing: "state = np.array([state])"
ACTION_SIZE = 2
# RNN
number_of_RNNmodels_reg = 5
number_of_RNNmodels_class = 7


def Agent_q_iteration(steps, env, model, target_model, iteration, current_state, mem_states, mem_actions, mem_rewards,
                      mem_terminal, mem_size, score, scores, true_score, true_scores, number_of_RNNmodels_class,
                      number_of_RNNmodels_reg, MODELS):
    """
    Do one iteration of acting then learning
    """
    epsilon = cartpole.get_epsilon_for_iteration(iteration)  # Choose epsilon based on the iteration
    start_state = current_state
    # Choose the action:
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = cartpole.choose_best_action(model, start_state)

    # Play one game iteration:
    next_state, reward, is_terminal, _ = env.step(action)
    steps += 1
    next_state = np.array([next_state])[0, :]  # Process state so that it's a numpy array, shape (4,)
    # Use RNN to predict reward
    predictions = []
    pre_class = []
    for k in range(number_of_RNNmodels_class):
        pred = MODELS[0][k].predict(np.array([next_state]))[0][0]
        pre_class.append(pred)
    class_pred = np.mean(pre_class)
    if class_pred >= 0.5:
        reward_pred = -100
    else:
        for j in range(number_of_RNNmodels_reg):
            prediction = MODELS[1][j].predict(np.array([next_state]))[0]
            predictions.append(prediction)
        reward_pred = np.mean(predictions)

    # get true reward
    if is_terminal:
        reward = -100
    else:
        reward = 2.4 - abs(next_state[0]) + 12 * 2 * math.pi / 360 - abs(next_state[2]) - abs(next_state[1])\
                 - abs(next_state[3])

    score += reward_pred
    true_score += reward

    # If DONE, reset model, modify reward, record score
    if is_terminal:
        env.reset()
        scores.append(score)  # Record score
        score = 0  # Reset score to zero
        true_scores.append(true_score)
        true_score = 0
    elif steps >= 200:
        env.reset()
        steps = 0
        scores.append(score)
        score = 0
        true_scores.append(true_score)
        true_score = 0

    cartpole.add_to_memory(
        iteration+1, mem_states, mem_actions, mem_rewards, mem_terminal, next_state, action, reward_pred, is_terminal)

    # Make then fit a batch (gamma=0.99, num_in_batch=32)
    number_in_batch = 32
    cartpole.make_n_fit_batch(model, target_model, 0.99, iteration,
                              mem_size, mem_states, mem_actions, mem_rewards, mem_terminal, number_in_batch)

    current_state = next_state

    return steps, action, reward_pred, is_terminal, epsilon, current_state, score, scores,true_score, true_scores



def Agent(t, ratio):
    """ Train the DQN to play Cartpole with RNN trained using data generated randomly
        """
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--num_rand_acts', help="Random actions before learning starts",
                        default=100, type=int)
    parser.add_argument('-m', '--mem_size', help="Size of the experience replay memory",
                        default=10 ** 4, type=int)
    args = parser.parse_args()

    # Set up logging:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Other things to modify
    number_training_steps = t
    print_progress_after = 10 ** 2
    Copy_model_after = 100

    number_random_actions = args.num_rand_acts
    mem_size = args.mem_size

    logger.info(' num_rand_acts = %s, mem_size = %s',
                number_random_actions, mem_size)

    # Make the model
    model = cartpole.make_model()
    model.summary()

    # Make the memories
    mem_states = cartpole.RingBufSimple(mem_size)
    mem_actions = cartpole.RingBufSimple(mem_size)
    mem_rewards = cartpole.RingBufSimple(mem_size)
    mem_terminal = cartpole.RingBufSimple(mem_size)

    print('Setting up Cartpole and pre-filling memory with random actions...')

    # Create and reset the Atari env:
    env = gym.make('CartPole-v1')
    env.reset()
    steps = 0


    # First make some random actions, and initially fill the memories with these:
    test_input = np.zeros((number_random_actions+1, 4))
    test_output = np.zeros((number_random_actions+1, 1))
    for i in range(number_random_actions + 1):
        iteration = i
        # Random action
        action = env.action_space.sample()
        next_state, reward, is_terminal, _ = env.step(action)
        steps += 1
        test_input[i] = next_state
        next_state = np.array([next_state])[0, :]  # Process state so that it's a numpy array, shape (4,)

        if is_terminal:
            reward = -100
            env.reset()

        else:
            reward = 2.4 - abs(next_state[0]) + 12 * 2 * math.pi / 360 - abs(next_state[2]) - abs(next_state[1]) \
                     - abs(next_state[3])

        if steps >= 200:
            env.reset()
            steps = 0

        test_output[i] = reward
        cartpole.add_to_memory(
            iteration, mem_states, mem_actions, mem_rewards, mem_terminal, next_state, action, reward, is_terminal)


    # Now do actions using the DQN, and train as we go...
    print('Finished the {} random actions...'.format(number_random_actions))
    current_state = next_state

    # For recroding the score
    score = 0
    scores = []
    true_score = 0
    true_scores = []
    train_number = 0
    test_number = 0
    train_input_class = list()
    train_output_class = list()
    train_input_reg = list()
    train_output_reg = list()

    # Create RNN
    model_class = []
    model_reg = []
    for i in range(number_of_RNNmodels_class):
        globals()['RNNmodel_{}'.format(i + 1)] = RNN.class_model()
        model_class.append(globals()['RNNmodel_{}'.format(i + 1)])
    model_class = RNN.check_model(model_class, number_of_RNNmodels_class, test_input)
    for i in range(number_of_RNNmodels_reg):
        mmodel = RNN.reg_model()
        model_reg.append(mmodel)
    model_reg = RNN.check_model(model_reg, number_of_RNNmodels_reg, test_input)
    MODELS = [model_class, model_reg]


    plt.ion()
    fig = plt.figure('Agent_r')
    for i in range(number_training_steps):

        iteration = number_random_actions + i

        # Copy model periodically and fit to this: this makes the learning more stable
        if i % Copy_model_after == 0:
            target_model = keras.models.clone_model(model)
            target_model.set_weights(model.get_weights())

        ret = random.random()
        if ret < ratio:
            train_number += 1
            steps, action, reward, is_terminal, epsilon, current_state, true_score, true_scores,train_input_class = cartpole.q_iteration(
                steps, env, model, target_model, iteration, current_state,
                mem_states, mem_actions, mem_rewards, mem_terminal, mem_size, true_score, true_scores,train_input_class)

            train_output_class.append(is_terminal)
            if not is_terminal:
                train_input_reg.append(current_state)
                train_output_reg.append(reward)
        else:
            test_number += 1
            steps, action, reward_pred, is_terminal, epsilon, current_state, score, scores,true_score, true_scores = \
                Agent_q_iteration(steps, env, model, target_model, iteration, current_state, mem_states, mem_actions,
                                          mem_rewards, mem_terminal, mem_size, score, scores, true_score, true_scores,
                                  number_of_RNNmodels_class, number_of_RNNmodels_reg, MODELS)

        # Print progress, time, and SAVE the model
        if (i + 1) % print_progress_after == 0:
            print('Training steps done: {}, Epsilon: {}'.format(i + 1, epsilon))
            print('Mean score = {}'.format(np.mean(scores)))
            print('Average scores for last 100 trials = {}'.format(np.mean(true_scores[::-1][0:100])))
            print('Ratio = {}'.format(train_number/(train_number+test_number)))
            # Test_acc = np.zeros((number_of_RNNmodels, 1))
            # for j in range(number_of_RNNmodels):
            #     test_acc, test_loss = rnnModel.test_RNNmodel(test_input, test_output, MODELS[j])
            #     Test_acc[j] = test_acc
            # print('RNN Test mean accuracy:', np.mean(Test_acc))

            plt.clf()
            plt.plot(true_scores)
            plt.ylabel('scores')
            plt.xlabel('Steps until {}'.format(i + 1))
            plt.pause(0.1)

        if len(train_input_class) == 100:
            for j in range(number_of_RNNmodels_class):
                hos = MODELS[0][j].fit(np.array(train_input_class), np.array(train_output_class), batch_size=100, epochs=20, verbose=0)
            train_input_class = list()
            train_output_class = list()
        if len(train_input_reg) == 100:
            for j in range(number_of_RNNmodels_reg):
                MODELS[1][j] = RNN.train_RNNmodel(np.array(train_input_reg), np.array(train_output_reg), MODELS[1][j])
            train_input_reg = list()
            train_output_reg = list()
    plt.ioff()
    # if len(training_input) > 0:
    #     for j in range(number_of_RNNmodels):
    #         MODELS[j] = RNN_reg.train_RNNmodel(np.array(training_input), np.array(training_output),MODELS[j])

    #   Save Agent_r
    file_name = os.path.join('Agents_0.06', 'Agent_r_5')
    model.save_weights(file_name)
    print('Agent_r saved')
    return scores, true_scores

