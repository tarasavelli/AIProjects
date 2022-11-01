import gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =   20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999



def default_Q_value():
    return 0


if __name__ == "__main__":



    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v0")
    env.seed(1)
    env.action_space.np_random.seed(1)


    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    episode_reward_record = deque(maxlen=100)
    


    for i in range(EPISODES):
        episode_reward = 0

        state = env.reset() ## making it such that we always start at the beginning point

        done = False

        while done == False:

            ## now must choose the next action (left, right, down, up)

            action = 0

            randval = np.random.uniform(0, 1)


            if randval < EPSILON:
                action = env.action_space.sample()
            else:
                stateRow = Q[state, :]
                action = np.argmax(stateRow)
            
            ## Will now go to the state specified by the action using env.step()

            newState, reward, finished, info = env.step(action)
            prediction = Q[state, action]

            if finished:
                Q[state, action] = prediction + LEARNING_RATE * (reward - prediction)
                Q_table.update({(state, action) : Q[state][action]})

            else:
                Q[state, action] = prediction + LEARNING_RATE * (reward + (DISCOUNT_FACTOR * np.max(Q[newState, :])) - prediction)
                Q_table.update({(state, action) : Q[state][action]})

            episode_reward +=  reward   
            state = newState
            done = finished

        
        EPSILON *= EPSILON_DECAY  

        episode_reward_record.append(episode_reward)



        
        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
        
            


    ####DO NOT MODIFY######
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    #######################







