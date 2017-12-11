# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 21:42:07 2017

@author: thy1995
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 17:00:03 2017

@author: FryingNemo
"""
import numpy as np
import random

class Dependent_bandit():
    def __init__(self,difficulty, ml):
        self.num_actions = 2
        self.difficulty = difficulty
        self.ml = ml
        self.reset()
        
    def set_restless_prob(self):
        self.bandit = np.array([self.restless_list[self.timestep],1 - self.restless_list[self.timestep]])
        
    def reset(self):
        self.timestep = 0
        if self.difficulty == "fast":
            self.variance = [0.1,self.ml]
        if self.difficulty == "medium":
            self.variance = [0.05,self.ml]
        if self.difficulty == "slow":
            self.variance = [0.025, self.ml]
        if self.difficulty == "normal":
            self.variance = [0.0125, self.ml]
        self.restless_list = self.sample()

    def sample(self):
#        temp = np.linspace(0,1, round(self.variance[1]/2))
#        if self.difficulty != "normal":
#            sample = np.tile(np.append(temp, temp[::-1]), self.variance[0])
#        else:
#            sample = [0.6] * (self.variance[0] * self.variance[1])
        sample  = []
        current = 0
        for i in range(self.variance[1]):
            switch = np.random.choice([True, False], p = [self.variance[0], 1 - self.variance[0]] )
            if switch:
                if current == 0:
                    current  = 1
                else:
                    current =0
            sample.append(current)
        return sample 
    
    def pullArm(self,action):
        #Get a random number.
        self.set_restless_prob()
        self.timestep += 1
        bandit = self.bandit[action]
        result = np.random.uniform()
        #result = 0.5
        if result < bandit:
            #return a positive reward.
            reward = 1
        else:
            #return a negative reward.
            reward = 0
        if self.timestep > self.ml: 
            done = True
        else: done = False
        return reward,done,self.timestep
    
def make_states(iterations, map_length):
    to_return = []
    possible_states = [[0,0], [0,1], [1,0], [1,1]]
    for _  in range(iterations):
        present_states = random.choice(possible_states)
        for i in range(map_length):
            to_return.append(present_states)
    return to_return
        
    