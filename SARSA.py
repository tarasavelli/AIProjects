from collections import deque
import gym
import random
import numpy as np
import time
import pickle

from collections import defaultdict


EPISODES =   20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999



def default_Q_value():
    return 0


def get_action(env, state):
    action = 0
    randval = np.random.uniform(0, 1)
    if randval < EPSILON:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    
    return action



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
       
        #TODO perform SARSA learning

        state = env.reset()

        done = False

        while done == False:

            action = get_action(env, state)
            
            ## getting the second action for the update
            state2, reward, finished, info = env.step(action)
            action2 = get_action(env, state2)

            

            prediction = Q[state][action]
            target = reward + DISCOUNT_FACTOR * Q[state2, action2]

            ## if reached terminal state, process is same as q learning
            if finished:
                Q[state, action] = prediction + LEARNING_RATE * (reward - prediction)
                Q_table.update({(state, action) : Q[state][action]})
            else:
                ## if not finished, process differs by not ALWAYS choosing the best action. Instead, we go through the epsilon greedy processa again to choose the next action
                Q[state][action] = prediction + LEARNING_RATE * (target - prediction)
                Q_table.update({(state, action) : Q[state][action]})


            episode_reward += reward
            state = state2
            done = finished
        
        
        EPSILON *= EPSILON_DECAY

        episode_reward_record.append(episode_reward)



            





        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    ####DO NOT MODIFY######
    model_file = open('SARSA_Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    #######################



