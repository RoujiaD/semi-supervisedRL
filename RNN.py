from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import regularizers


STATE_SHAPE = (4,)
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

def class_model():
    model = Sequential()
    model.add(Dense(32, input_shape=STATE_SHAPE, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def reg_model():
    model = Sequential()
    model.add(Dense(64, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), input_shape=STATE_SHAPE, activation="relu"))
    model.add(Dense(256, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), input_shape=STATE_SHAPE,
              activation="relu"))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer=Adam(lr=LEARNING_RATE), metrics=['mae'])
    return model



def check_model(MODELS, number_of_models, test_input):
    ALLpred = list()
    done = True
    for i in range(number_of_models):
        preds = []
        for j in range(10):
            pred = MODELS[i].predict(np.array([test_input[j]]))
            preds.append(pred[0][0])
        ALLpred.append(preds)
        print('Initial prediction {}'.format(i + 1), preds)
    for i in range(number_of_models):
        for j in range(i + 1, number_of_models):
            count = 0
            for k in range(len(preds)):
                if abs(ALLpred[i][k] - ALLpred[j][k]) < 0.005:
                    count += 1
            if count == len(preds):
                done = False
                MODELS[j] = reg_model()
    if not done:
        check_model(MODELS, number_of_models, test_input)
    else:
        print('Model checking is done.')
    return MODELS



def norm(state):
    x = list()
    x_dot = list()
    theta = list()
    theta_dot = list()
    for i in range(len(state)):
        x.append(state[i][0])
        x_dot.append(state[i][1])
        theta.append(state[i][2])
        theta_dot.append(state[i][3])
    whole = [x, x_dot, theta, theta_dot]
    for i in range(len(whole)):
        mean = np.mean(whole[i])
        M = np.max(whole[i])
        m = np.min(whole[i])
        for j in range(len(whole[i])):
            whole[i][j] = (whole[i][j] - mean)/abs((np.max(whole[i]) - np.min(whole[i])))
    for i in range(len(state)):
        for j in range(len(state[0])):
            state[i][j] = whole[j][i]
    return state


def train_RNNmodel(train_states, train_rewards, model):
    train_states = norm(train_states)
    # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(train_states, train_rewards, epochs=EPOCHS, verbose=0, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT)
    return model

