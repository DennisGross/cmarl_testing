from pettingzoo import ParallelEnv
import random
import pygame
import sys
from gymnasium import spaces
import gymnasium
import time
import numpy as np
class CoinGame(ParallelEnv):
    metadata = {
        "name": "coin_game",
    }

    def __init__(self):
        self.MAX_X = 5
        self.MAX_Y = 5
        self.MAX_FUEL = 10
        self.ENERGY_STATION_X = 2
        self.ENERGY_STATION_Y = 2
        self.MAX_STEPS = 50
        self.current_step = 0
        self.possible_agents = ["agent_0", "agent_1"]#, "agent_2"]
        self.agents = self.possible_agents.copy()

        self.create_world()
        
        self.render_mode = ""

        self.observation_spaces = {
            "agent_0": spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
            "agent_1": spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
            #"agent_2": spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32),
        }

        self.action_spaces = {
            "agent_0": gymnasium.spaces.discrete.Discrete(4),
            "agent_1": gymnasium.spaces.discrete.Discrete(4),
            #"agent_2": gymnasium.spaces.discrete.Discrete(4)
        }

    def create_world(self):
        self.current_step = 0

        self.x0 = random.randint(0, self.MAX_X)
        self.y0 = random.randint(0, self.MAX_Y)
        self.fuel0 = self.MAX_FUEL

        self.x1 = random.randint(0, self.MAX_X)
        self.y1 = random.randint(0, self.MAX_Y)
        self.fuel1 = self.MAX_FUEL

        #self.x2 = random.randint(0, self.MAX_X)
        #self.y2 = random.randint(0, self.MAX_Y)
        #self.fuel2 = self.MAX_FUEL

        self.place_coins()

    def place_coins(self):
        self.place_coin0()
        self.place_coin1()
        self.place_coin2()

    def place_coin0(self):
        # Make sure the coins are not in the same location as the agents
        self.cx0 = random.randint(0, self.MAX_X)
        self.cy0 = random.randint(0, self.MAX_Y)
        while self.cx0 == self.x0 and self.cy0 == self.y0 or self.cx0 == self.x1 and self.cy0 == self.y1:# or self.cx0 == self.x2 and self.cy0 == self.y2:
            self.cx0 = random.randint(0, self.MAX_X)
            self.cy0 = random.randint(0, self.MAX_Y)
            
        
    def place_coin1(self):
        # Make sure the coins are not in the same location as the agents
        self.cx1 = random.randint(0, self.MAX_X)
        self.cy1 = random.randint(0, self.MAX_Y)
        while self.cx1 == self.x0 and self.cy1 == self.y0 or self.cx1 == self.x1 and self.cy1 == self.y1:# or self.cx1 == self.x2 and self.cy1 == self.y2:
            self.cx1 = random.randint(0, self.MAX_X)
            self.cy1 = random.randint(0, self.MAX_Y)
    
    def place_coin2(self):
        # Make sure the coins are not in the same location as the agents
        self.cx2 = random.randint(0, self.MAX_X)
        self.cy2 = random.randint(0, self.MAX_Y)
        while self.cx2 == self.x0 and self.cy2 == self.y0 or self.cx2 == self.x1 and self.cy2 == self.y1:# or self.cx2 == self.x2 and self.cy2 == self.y2:
            self.cx2 = random.randint(0, self.MAX_X)
            self.cy2 = random.randint(0, self.MAX_Y)

    


    def reset(self, seed=None, options=None):
        # Reset the environment
        self.create_world()

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.possible_agents}

        return self.get_observations(), infos


    def get_observations(self):
        observations = {
            "agent_0": (
                self.x0/self.MAX_X,
                self.y0/self.MAX_Y,
                self.fuel0/self.MAX_FUEL,
                self.x1/self.MAX_X,
                self.y1/self.MAX_Y,
                self.fuel1/self.MAX_FUEL,
                self.cx0/self.MAX_X,
                self.cy0/self.MAX_Y,
                self.cx1/self.MAX_X,
                self.cy1/self.MAX_Y,
                self.cx2/self.MAX_X,
                self.cy2/self.MAX_Y
            ),
            "agent_1": (
                self.x1/self.MAX_X,
                self.y1/self.MAX_Y,
                self.fuel1/self.MAX_FUEL,
                self.x0/self.MAX_X,
                self.y0/self.MAX_Y,
                self.fuel0/self.MAX_FUEL,
                self.cx0/self.MAX_X,
                self.cy0/self.MAX_Y,
                self.cx1/self.MAX_X,
                self.cy1/self.MAX_Y,
                self.cx2/self.MAX_X,
                self.cy2/self.MAX_Y
            )
        }
        return observations


    def step(self, actions):
        # Get actions
        agent0_action = actions["agent_0"]
        agent1_action = actions["agent_1"]
        # Set initial reward per step
        current_reward = 0
        # Set done to False in the current step
        done = False

        # Move agent 0, decrease fuel, and check boundaries
        if agent0_action == 0:
            self.x0 = min(self.MAX_X, self.x0 + 1)
            self.fuel0 -= 1
        elif agent0_action == 1:
            self.x0 = max(0, self.x0 - 1)
            self.fuel0 -= 1
        elif agent0_action == 2:
            self.y0 = min(self.MAX_Y, self.y0 + 1)
            self.fuel0 -= 1
        elif agent0_action == 3:
            self.y0 = max(0, self.y0 - 1)
            self.fuel0 -= 1

        # Move agent 1, decrease fuel, and check boundaries
        if agent1_action == 0:
            self.x1 = min(self.MAX_X, self.x1 + 1)
            self.fuel1 -= 1
        elif agent1_action == 1:
            self.x1 = max(0, self.x1 - 1)
            self.fuel1 -= 1
        elif agent1_action == 2:
            self.y1 = min(self.MAX_Y, self.y1 + 1)
            self.fuel1 -= 1
        elif agent1_action == 3:
            self.y1 = max(0, self.y1 - 1)
            self.fuel1 -= 1
        
        

        # Check if on energy station
        if self.x0 == self.ENERGY_STATION_X and self.y0 == self.ENERGY_STATION_Y:
            if self.fuel0 < self.MAX_FUEL:
                current_reward += 0.1
            self.fuel0 = self.MAX_FUEL
        if self.x1 == self.ENERGY_STATION_X and self.y1 == self.ENERGY_STATION_Y:
            if self.fuel1 < self.MAX_FUEL:
                current_reward += 0.1
            self.fuel1 = self.MAX_FUEL
        
        # Check if on agent is on coin0
        if (self.x0 == self.cx0 and self.y0 == self.cy0) or (self.x1 == self.cx0 and self.y1 == self.cy0):# or (self.x2 == self.cx0 and self.y2 == self.cy0):
            self.place_coin0()
            current_reward += 10
        
        # Check if on agent is on coin1
        if (self.x0 == self.cx1 and self.y0 == self.cy1) or (self.x1 == self.cx1 and self.y1 == self.cy1):# or (self.x2 == self.cx1 and self.y2 == self.cy1):
            self.place_coin1()
            current_reward += 10
        
        # Check if on agent is on coin2
        if (self.x0 == self.cx2 and self.y0 == self.cy2) or (self.x1 == self.cx2 and self.y1 == self.cy2):# or (self.x2 == self.cx2 and self.y2 == self.cy2):
            self.place_coin2()
            current_reward += 10
        
        # Check if fuel is empty
        if self.fuel0 == 0 or self.fuel1 == 0:# or self.fuel2 == 0:
            current_reward = 0
            done = True

        
        # Get observations
        observations = self.get_observations()

        # Rewards
        rewards = {
            "agent_0": current_reward,
            "agent_1": current_reward,
            #"agent_2": current_reward,
        }

        self.current_step += 1

        if self.current_step >= self.MAX_STEPS:
            done = True

        # Dones
        dones = {
            "agent_0": done,
            "agent_1": done,
            #"agent_2": done,
        }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.possible_agents}

        return observations, rewards, dones, dones, infos


    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]


if __name__ == "__main__":
    env = CoinGame()
    env.reset()
    for i in range(100):
        observations, rewards, dones, truncation, infos = env.step({"agent_0": random.randint(0,3), "agent_1": random.randint(0,3)})
        if all(dones.values()) and all(truncation.values()):
            print("Done")
            env.reset()