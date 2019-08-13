import os
import random
import logging

import gym
import numpy as np
import keras
import matplotlib.pyplot as plt
from DQN_reg import cartpole
from DQN_reg import RNN
import math


# State and action sizes *for this particular environment*. These are constants (fixed throughout), so USE_CAPS
STATE_SHAPE = (4,) # This is the shape after pre-processing: "state = np.array([state])"
ACTION_SIZE = 2
# RNN
number_of_RNNmodels_reg = 5
number_of_RNNmodels_class = 7
# number of samples used to train rnn in the beginning
number = 500


def show_max(list):
    index = 0
    max = 0
    for i in range(len(list)):
        count = 0
        for j in range(i+1, len(list)):
            if list[j] == list[i]:
                count +=1
        if count > max:
            max = count
            index = i
    return list[index]



def q_iteration(
        steps, env, model, target_model, iteration, current_state, mem_states, mem_actions, mem_rewards, mem_terminal,
        mem_size, score, scores, true_score, true_scores, MODELS, Ask_number, correct_pred, Ask_input_class,
        Ask_output_class, Ask_input_reg, Ask_output_reg, can_ask, t, ratio):
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
    next_state, _, is_terminal, _ = env.step(action)
    steps += 1
    next_state = np.array([next_state])[0, :]  # Process state so that it's a numpy array, shape (4,)

    # Use RNN to predict reward
    IfAsk = False
    predictions = []
    for j in range(number_of_RNNmodels_class):
        prediction = MODELS[0][j].predict(np.array([next_state]))[0][0]
        predictions.append(prediction)
    if np.mean(predictions) > 0.9:
        reward_pred = -100
    elif np.mean(predictions) < 0.1:
        predictions=[]
        for j in range(number_of_RNNmodels_reg):
            prediction = MODELS[1][j].predict(np.array([next_state]))
            prediction = prediction[0]

            predictions.append(prediction)
        if (t - iteration + 100) <= (round(t * ratio) - number - Ask_number):
            IfAsk = True

        if (np.max(predictions) - np.min(predictions)) < 0.1:
            reward_pred = np.mean(predictions)
        else:
            IfAsk = True
            reward_pred = None

    else:
        IfAsk = True

    # Retrain the RNNmodels
    if IfAsk:
        if can_ask:
            if is_terminal:
                reward_pred = -100
            else:
                reward_pred = 2.4 - abs(next_state[0]) + 12 * 2 * math.pi / 360 - abs(next_state[2]) \
                              - abs(next_state[1]) - abs(next_state[3])
                Ask_input_reg.append(next_state)
                Ask_output_reg.append(reward_pred)

            if abs(show_max(predictions) - reward_pred) < 0.1:
                correct_pred += 1

            Ask_input_class.append(next_state)
            Ask_output_class.append(is_terminal)
            Ask_number += 1
        else:
            reward_pred = np.mean(predictions)

    if is_terminal:
        reward = -100
    else:
        reward = 2.4 - abs(next_state[0]) + 12 * 2 * math.pi / 360 - abs(next_state[2]) \
                      - abs(next_state[1]) - abs(next_state[3])

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
        scores.append(score)
        score = 0
        steps = 0
        true_scores.append(true_score)
        true_score = 0
        env.reset()

    cartpole.add_to_memory(
        iteration+1, mem_states, mem_actions, mem_rewards, mem_terminal, next_state, action, reward_pred, is_terminal)

    # Make then fit a batch (gamma=0.99, num_in_batch=32)
    number_in_batch = 32
    cartpole.make_n_fit_batch(model, target_model, 0.99, iteration,
                              mem_size, mem_states, mem_actions, mem_rewards, mem_terminal, number_in_batch)

    current_state = next_state

    return steps, action, reward_pred, is_terminal, epsilon, current_state, score, scores, true_score, true_scores, Ask_number, correct_pred, Ask_input_class, Ask_output_class,Ask_input_reg, Ask_output_reg, can_ask



def Agent(t, ratio):
    """ Train the DQN to play Cartpole, and the agent will ask for reward if it's uncertain about the predictions
    """
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-r', '--num_rand_acts', help="Random actions before learning starts",
    #                     default = 100, type=int)
    # parser.add_argument('-m', '--mem_size', help="Size of the experience replay memory",
    #                     default = 10**4, type=int)
    # args = parser.parse_args()



    # Set up logging:
    logging.basicConfig(level=logging.INFO) # Is this in the right place?
    logger = logging.getLogger(__name__)

    # Other things to modify
    train_number = number
    number_testing_steps = t - train_number
    print_progress_after = 10**2
    Copy_model_after = 100

    number_random_actions = 100
    mem_size = 10000

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
    test_input = np.zeros((number_random_actions + 1, 4))
    # test_output = np.zeros((number_random_actions + 1, 1))

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

        # test_output[i] = reward
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
    train_input_class = list()
    train_output_class = list()
    train_input_reg = list()
    train_output_reg = list()


    plt.ion()
    fig = plt.figure('Agent_a')
    for i in range(train_number):

        iteration = number_random_actions + i

        # Copy model periodically and fit to this: this makes the learning more stable
        if i % Copy_model_after == 0:
            target_model = keras.models.clone_model(model)
            target_model.set_weights(model.get_weights())

        steps, action, reward, is_terminal, epsilon, current_state, true_score, true_scores, train_input_class = cartpole.q_iteration(
            steps, env, model, target_model, iteration, current_state,
            mem_states, mem_actions, mem_rewards, mem_terminal, mem_size, true_score, true_scores,train_input_class)

        train_output_class.append(is_terminal)
        if not is_terminal:
            train_input_reg.append(current_state)
            train_output_reg.append(reward)

        # Print progress, score, and SAVE the model
        if (i + 1) % print_progress_after == 0:
            print('Training steps done: {}, Epsilon: {}'.format(i + 1, epsilon))
            print('Mean score = {}'.format(np.mean(true_scores)))
            print('Average scores for last 100 trials = {}'.format(np.mean(true_scores[::-1][0:100])))
            plt.clf()
            plt.plot(true_scores)
            plt.ylabel('scores')
            plt.xlabel('Steps until {}'.format(number_random_actions + i + 1))
            plt.pause(0.1)

    # Create & Train RNN
    model_class=[]
    model_reg=[]
    for i in range(number_of_RNNmodels_class):
        globals()['RNNmodel_{}'.format(i + 1)] = RNN.class_model()
        model_class.append(globals()['RNNmodel_{}'.format(i + 1)])
    model_class = RNN.check_model(model_class, number_of_RNNmodels_class, test_input)
    for i in range(number_of_RNNmodels_reg):
        mmodel = RNN.reg_model()
        model_reg.append(mmodel)
    model_reg = RNN.check_model(model_reg, number_of_RNNmodels_reg, test_input)
    MODELS = [model_class,model_reg]

    for i in range(number_of_RNNmodels_class):
        his = MODELS[0][i].fit(np.array(train_input_class), np.array(train_output_class),batch_size=100, epochs=20, verbose=0)

    for i in range(number_of_RNNmodels_reg):
        MODELS[1][i]= RNN.train_RNNmodel(np.array(train_input_reg), np.array(train_output_reg), MODELS[1][i])



    # Now use RNNs:
    env.reset()
    steps = 0
    can_ask = True
    Ask_number = 0
    correct_pred = 0
    Ask_input_class = list()
    Ask_output_class = list()
    Ask_input_reg = []
    Ask_output_reg = []

    for i in range(number_testing_steps):
        iteration = train_number + number_random_actions + i

        # Copy model periodically and fit to this: this makes the learning more stable
        if i % Copy_model_after == 0:
            target_model = keras.models.clone_model(model)
            target_model.set_weights(model.get_weights())

        steps, action, reward_pred, is_terminal, epsilon, current_state, score, scores,true_score, true_scores, Ask_number, correct_pred, \
        Ask_input_class, Ask_output_class,Ask_input_reg, Ask_output_reg, can_ask= q_iteration(
            steps, env, model, target_model, iteration, current_state,mem_states, mem_actions, mem_rewards, mem_terminal,
            mem_size, score, scores,true_score, true_scores, MODELS, Ask_number, correct_pred, Ask_input_class,
            Ask_output_class, Ask_input_reg, Ask_output_reg, can_ask, t, ratio)



        # Print progress, time, and SAVE the model
        if (i + 1) % print_progress_after == 0:
            print('Training steps done: {}, Epsilon: {}'.format(train_number + i + 1, epsilon))
            print('Mean score = {}'.format(np.mean(true_scores)))
            print('Average scores for last 100 trials = {}'.format(np.mean(true_scores[::-1][0:100])))
            if Ask_number != 0:
                print('Ask_number = {}, Accuracy = {}'.format(Ask_number, correct_pred/Ask_number))
            # Test_acc = np.zeros((number_of_RNNmodels,1))
            # for j in range(number_of_RNNmodels):
            #     test_acc, test_loss = rnnModel.test_RNNmodel(test_input,test_output,MODELS[j])
            #     Test_acc[j] = test_acc
            # print('RNN Test mean accuracy:', np.mean(Test_acc))
            plt.clf()
            plt.plot(true_scores)
            plt.ylabel('scores')
            plt.xlabel('Steps until {}'.format(train_number + i + 1))
            plt.pause(0.1)
        if Ask_number >= (round(t * ratio) - train_number):
            can_ask = False

        if len(Ask_input_class) == 100:
            for j in range(number_of_RNNmodels_class):
                his= MODELS[0][j].fit(np.array(Ask_input_class), np.array(Ask_output_class),batch_size=100, epochs=20, verbose=0)
            Ask_input_class = list()
            Ask_output_class = list()
        if len(Ask_input_reg) == 100:
            for j in range(number_of_RNNmodels_reg):
                MODELS[1][j] = RNN.train_RNNmodel(np.array(Ask_input_reg), np.array(Ask_output_reg), MODELS[1][j])
            Ask_input_reg = list()
            Ask_output_reg = list()


    plt.ioff()
    # if len(Ask_output) > 0:
    #     for j in range(number_of_RNNmodels):
    #         MODELS[j]= RNN_reg.train_RNNmodel(
    #             np.array(Ask_input), np.array(Ask_output), MODELS[j])

    # Save Agent_a
    file_name = os.path.join('Agents_0.06', 'Agent_a_1')
    model.save_weights(file_name)
    print('Agent_a saved')
    ratio = (train_number + Ask_number)/t
    print('Ration = {}'.format(ratio))

    return scores, true_scores
