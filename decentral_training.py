import argparse
from logger import *
from pettingzoo.mpe import simple_speaker_listener_v4

import env_stuff.coin_collector
import env_stuff.genetic_coin_collector
import env_stuff.knight_wizard_zombies
import env_stuff.genetic_knight_wizard_zombies
import env_stuff.genetic_simple_spread

import gc

import glob
import os
import time

import supersuit as ss
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
import numpy as np


from pettingzoo.mpe import simple_spread_v3
import random

import torch

import itertools
import math

import os
import numpy as np
import torch
from typing import List
from collections import OrderedDict, deque
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import math
# Import the PettingZoo simple_spread environment
from pettingzoo.mpe import simple_spread_v3, simple_speaker_listener_v4

# Define the device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sum_distances(scalars):
    return sum(abs(a - b) for a, b in itertools.combinations(scalars, 2))


class ReplayBuffer(object):
    def __init__(self, max_size: int, state_dimension: int):
        self.size = max_size
        self.memory_counter = 0
        self.state_memory = np.zeros((self.size, state_dimension), dtype=np.float32)
        self.new_state_memory = np.zeros((self.size, state_dimension), dtype=np.float32)
        self.action_memory = np.zeros(self.size, dtype=np.int64)
        self.reward_memory = np.zeros(self.size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.size, dtype=np.bool_)

    def store_transition(self, state: np.array, action: int, reward: float, state_: np.array, done: bool):
        index = self.memory_counter % self.size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.memory_counter += 1

    def sample(self, batch_size: int) -> tuple:
        max_mem = min(self.memory_counter, self.size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        return (torch.tensor(states).to(DEVICE),
                torch.tensor(actions).to(DEVICE),
                torch.tensor(rewards).to(DEVICE),
                torch.tensor(states_).to(DEVICE),
                torch.tensor(terminal).to(DEVICE))

class DeepQNetwork(nn.Module):
    def __init__(self, state_dimension: int, number_of_neurons: List[int], number_of_actions: int, lr: float):
        super(DeepQNetwork, self).__init__()
        layers = OrderedDict()
        previous_neurons = state_dimension
        for i in range(len(number_of_neurons)+1):
            if i == len(number_of_neurons):
                layers[str(i)] = torch.nn.Linear(previous_neurons, number_of_actions)
            else:
                layers[str(i)] = torch.nn.Linear(previous_neurons, number_of_neurons[i])
                previous_neurons = number_of_neurons[i]
        self.layers = torch.nn.Sequential(layers)
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.DEVICE = DEVICE
        self.to(self.DEVICE)

    def forward(self, state: np.array) -> int:
        try:
            x = state
            for i in range(len(self.layers)):
                if i == (len(self.layers)-1):
                    x = self.layers[i](x)
                else:
                    x = F.relu(self.layers[i](x))
            return x
        except:
            state = torch.tensor(state).float().to(DEVICE)
            x = state
            for i in range(len(self.layers)):
                if i == (len(self.layers)-1):
                    x = self.layers[i](x)
                else:
                    x = F.relu(self.layers[i](x))
            return x

    def save_checkpoint(self, file_name: str):
        torch.save(self.state_dict(), file_name)

    def load_checkpoint(self, file_name: str):
        self.load_state_dict(torch.load(file_name))

class DQNAgent():
    def __init__(self, state_dimension, number_of_neurons, number_of_actions, epsilon=1, epsilon_dec=0.99999, epsilon_min=0.1, gamma=0.99, learning_rate=0.001, replace=100, batch_size=64, replay_buffer_size=10000):
        self.number_of_actions = number_of_actions
        self.replay_buffer = ReplayBuffer(replay_buffer_size, state_dimension)
        self.q_eval = DeepQNetwork(state_dimension, number_of_neurons, number_of_actions, learning_rate)
        self.q_next = DeepQNetwork(state_dimension, number_of_neurons, number_of_actions, learning_rate)
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.gamma = torch.tensor(gamma).to(DEVICE)
        self.replace = replace
        self.batch_size = batch_size
        self.exp_counter = 0
        self.learn_step_counter = 0

    def save(self, path: str):
        try:
            os.mkdir(path)
        except Exception as msg:
            pass
        self.q_eval.save_checkpoint(os.path.join(path, 'q_eval.chkpt'))
        self.q_next.save_checkpoint(os.path.join(path, 'q_next.chkpt'))

    def load(self, path: str):
        try:
            self.q_eval.load_checkpoint(os.path.join(path, 'q_eval.chkpt'))
            self.q_next.load_checkpoint(os.path.join(path, 'q_next.chkpt'))
        except Exception as msg:
            print(msg)

    def store_experience(self, state: np.array, action: int, reward: float, n_state: np.array, done: bool):
        self.replay_buffer.store_transition(state, action, reward, n_state, done)
        self.exp_counter += 1

    def select_action(self, state: np.ndarray, deploy=False) -> int:
        if deploy:
            return int(torch.argmax(self.q_eval.forward(state)).item())
        if torch.rand(1).item() < self.epsilon:
            self.epsilon *= self.epsilon_dec
            self.epsilon = max(self.epsilon, self.epsilon_min)
            return int(torch.randint(0, self.number_of_actions, (1,)).item())
        else:
            return int(torch.argmax(self.q_eval.forward(state)).item())

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return self.q_eval.forward(state).detach().cpu()

    def replace_target_network(self):
        if self.learn_step_counter % self.replace == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def step_learn(self):
        if self.exp_counter < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        state_batch, action_batch, reward_batch, n_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        indices = torch.arange(0, self.batch_size).long()
        action_batch = action_batch.long()
        q_pred = self.q_eval.forward(state_batch)[indices, action_batch]
        q_next = self.q_next.forward(n_state_batch).max(dim=1).values.to(DEVICE)
        q_next[done_batch] = 0
        q_target = reward_batch.to(DEVICE) + self.gamma * q_next
        loss = self.q_eval.loss(q_target, q_pred)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

def dec(env, agents, num_episodes, deploy, top_percentage=0.9, low_percentage=0.1, model_path="", training_steps=None, gtest=None):
    best_avg_score = -np.inf  # Initialize best average score
    score_history = deque(maxlen=100)  # To store the last 100 total rewards
    all_rewards = []
    all_q_distances = []
    all_max_q_values = []
    all_training_steps = 0
    for episode in range(1, num_episodes + 1):
        observations, info = env.reset()
        episode_rewards = {agent_id: 0 for agent_id in env.agents}
        if deploy:
            max_q_values = []
            for agent_id, obs in observations.items():
                q_values = agents[agent_id].q_values(obs)
                max_q_values.append(torch.max(q_values).item())
            all_q_distances.append(sum_distances(max_q_values))
            all_max_q_values.extend(max_q_values)


        while True:
            actions = {}
            # Each agent selects an action based on its current observation
            for agent_id, obs in observations.items():
                action = agents[agent_id].select_action(obs, deploy)
                actions[agent_id] = action

            # Environment steps with the selected actions
            next_observations, rewards, dones, truncations, infos = env.step(actions)

            # Store experiences and learn for each agent
            for agent_id in env.agents:
                agents[agent_id].store_experience(state=observations[agent_id],
                                                  action=actions[agent_id],
                                                  reward=rewards[agent_id],
                                                  n_state=next_observations[agent_id],
                                                  done=dones[agent_id])
                if not deploy:
                    agents[agent_id].step_learn()
                episode_rewards[agent_id] += rewards[agent_id]

            observations = next_observations

            all_training_steps += 1

            
            if training_steps is not None and all_training_steps >= training_steps:
                break

            # If all agents are done, break the loop
            if all(dones.values()) or all(truncations.values()):
                break

        # Calculate total reward for the episode
        total_episode_reward = sum(episode_rewards.values())
        
        if deploy:
            if math.isnan(total_episode_reward) == False:
                all_rewards.append(total_episode_reward)
                if gtest:
                    gtest.set_individual_fitness(total_episode_reward)
            else:
                if gtest:
                    gtest.set_individual_fitness(math.inf)
        score_history.append(total_episode_reward)
        avg_score = np.mean(score_history)

        # Check if this is the best average score over the last 100 episodes
        if avg_score > best_avg_score and len(score_history) == 100:
            best_avg_score = avg_score
            # Save the models
            if deploy == False:
                # Make sure that model_path exists
                try:
                    os.mkdir(model_path)
                except Exception as msg:
                    pass
                for agent_id in agents:
                    agents[agent_id].save(f'{model_path}/best_model_agent_{agent_id}')
            #print(f'Episode {episode}, New Best Average Reward over 100 Episodes: {best_avg_score:.2f}')

        if training_steps is not None and all_training_steps >= training_steps:
            break

    

    if deploy:
        # Get the top 10% quantile of q_distances
        top_q_distance = np.quantile(all_q_distances, top_percentage)

        low_q_value = np.quantile(all_max_q_values, low_percentage)
        # avg_reward, rewards, top_q_distances_min, low_q_value 
        return sum(all_rewards)/len(all_rewards), all_rewards, top_q_distance, low_q_value


def dec_q_testing(env, agents, num_episodes, top_q_distances_min, all_rewards, gtest=None):
    all_q_distances = []
    all_max_q_values = []
    all_training_steps = 0
    for episode in range(1, num_episodes + 1):
        observations, info = env.reset()
        episode_rewards = {agent_id: 0 for agent_id in env.agents}
        print("====================================")
        max_q_values = []
        for agent_id, obs in observations.items():
            q_values = agents[agent_id].q_values(obs)
            print(q_values)
            max_q_values.append(torch.max(q_values).item())

        if sum_distances(max_q_values)<top_q_distances_min:
            if gtest:
                gtest.set_individual_fitness(math.inf)
            continue
        print(sum_distances(max_q_values), top_q_distances_min)
        while True:
            actions = {}
            # Each agent selects an action based on its current observation
            for agent_id, obs in observations.items():
                action = agents[agent_id].select_action(obs, True)
                actions[agent_id] = action

            # Environment steps with the selected actions
            next_observations, rewards, dones, truncations, infos = env.step(actions)

            # Rewards
            for agent_id in env.agents:
                episode_rewards[agent_id] += rewards[agent_id]

            observations = next_observations


            # If all agents are done, break the loop
            if all(dones.values()):
                break

        
        # Calculate total reward for the episode
        total_episode_reward = sum(episode_rewards.values())
        print(total_episode_reward)
        if math.isnan(total_episode_reward) == False:
            all_rewards.append(total_episode_reward)
            if gtest:
                gtest.set_individual_fitness(total_episode_reward)
        else:
            if gtest:
                gtest.set_individual_fitness(math.inf)
       


    return sum(all_rewards)/len(all_rewards), all_rewards


def dec_q_low_testing(env, agents, num_episodes, q_value, all_rewards):
    all_q_distances = []
    all_max_q_values = []
    all_training_steps = 0
    episode = 1
    while episode < num_episodes+1:
        observations, info = env.reset()
        episode_rewards = {agent_id: 0 for agent_id in env.agents}
        
        max_q_values = []
        for agent_id, obs in observations.items():
            q_values = agents[agent_id].q_values(obs)
            max_q_values.append(torch.max(q_values).item())

        if all(q_value > x for x in max_q_values):
            pass
        else:
            continue

        while True:
            actions = {}
            # Each agent selects an action based on its current observation
            for agent_id, obs in observations.items():
                action = agents[agent_id].select_action(obs, True)
                actions[agent_id] = action

            # Environment steps with the selected actions
            next_observations, rewards, dones, truncations, infos = env.step(actions)

            # Rewards
            for agent_id in env.agents:
                episode_rewards[agent_id] += rewards[agent_id]

            observations = next_observations


            # If all agents are done, break the loop
            if all(dones.values()):
                break
        episode += 1

            
        # Calculate total reward for the episode
        total_episode_reward = sum(episode_rewards.values())
        if math.isnan(total_episode_reward) == False:
            all_rewards.append(total_episode_reward)

       


    return sum(all_rewards)/len(all_rewards), all_rewards





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="simple_spread/model.zip")
    parser.add_argument("--env_name", type=str, default="simple_spread_v3")
    parser.add_argument("--N_agents", type=int, default=2)
    parser.add_argument("--local_ratio", type=float, default=0.5)
    parser.add_argument("--train", type=int, default=0)
    parser.add_argument("--training_steps", type=int, default=100)
    parser.add_argument("--test_samples", type=int, default=2)
    parser.add_argument("--test_budget", type=int, default=25)
    parser.add_argument("--calibration_number", type=int, default=10)
    parser.add_argument("--top_percentage", type=float, default=0.9)
    parser.add_argument("--num_cpus", type=int, default=20)
    parser.add_argument("--num_instances", type=int, default=256)
    # To dict
    args_dict = vars(parser.parse_args())
    return args_dict

def build_dec_env(env_name, N_agents, local_ratio, is_gtest=False):
    env = None
    gtest = None
    if env_name == "simple_spread_v3":
        env = simple_spread_v3.parallel_env(N=N_agents, local_ratio=local_ratio)
        env.reset()
        if is_gtest:
            gtest = env_stuff.genetic_simple_spread.GGTest(env, 200, N_agents=N_agents)
    elif env_name == "simple_speaker_listener_v4":
        env = simple_speaker_listener_v4.parallel_env()
        env.reset()
        if is_gtest:
            gtest = env_stuff.genetic_simple_spread.GGTest(env, 200, N_agents=N_agents)
    elif env_name == "knight_wizard_zombies":
        env = env_stuff.knight_wizard_zombies.KnightWizardZombies()
        env.reset()
        if is_gtest:
            gtest = env_stuff.genetic_knight_wizard_zombies.GGTest(env, 200)
    elif env_name == "coin_collector":
        env = env_stuff.coin_collector.CoinGame()
        env.reset()
        if is_gtest:
            gtest = env_stuff.genetic_coin_collector.GGTest(env, 200)
    return env, gtest

if __name__ == "__main__":
    args = get_args()
    low_percentage = 0.75
    print(args)
    
    model_path = args["model_path"]
    # Dec = Decentralized
    env, _ = build_dec_env(args["env_name"], args["N_agents"], args["local_ratio"])
    # Get the number of agents and their action/observation spaces
    num_agents = len(env.agents)
    obs_shape = env.observation_space(env.agents[0]).shape[0]
    num_actions = env.action_space(env.agents[0]).n

    # Define hyperparameters
    state_dimension = obs_shape
    number_of_neurons = [64, 64]  # Two hidden layers with 64 neurons each
    epsilon = 1.0
    epsilon_dec = 0.9999
    epsilon_min = 0.1
    gamma = 0.95
    learning_rate = 0.001
    replace = 1000
    batch_size = 64
    replay_buffer_size = 100000

    # Create a DQNAgent for each agent in the environment
    agents = {}
    for idx, agent_id in enumerate(env.agents):
        state_dimension = env.observation_space(agent_id).shape[0]
        agents[agent_id] = DQNAgent(state_dimension=state_dimension,
                                    number_of_neurons=number_of_neurons,
                                    number_of_actions=num_actions,
                                    epsilon=epsilon,
                                    epsilon_dec=epsilon_dec,
                                    epsilon_min=epsilon_min,
                                    gamma=gamma,
                                    learning_rate=learning_rate,
                                    replace=replace,
                                    batch_size=batch_size,
                                    replay_buffer_size=replay_buffer_size)
        # Load the best model if it exists
        if os.path.exists(f'{model_path}/best_model_agent_{agent_id}'):
            agents[agent_id].load(f'{model_path}/best_model_agent_{agent_id}')

    if args["train"]:
        print("Training")
        # Train on training steps not on episodes (therefore passing training steps)
        dec(env, agents, args["training_steps"], deploy=False, model_path=model_path, training_steps=args["training_steps"])
    else:
        m_logger = Logger("logs/dec_logs.csv", args["env_name"], args["N_agents"],args["local_ratio"],args["model_path"], args["calibration_number"], args["test_budget"], args["top_percentage"])
        # Random Testing
        if False:
            print("Random Testing")
            testing_times = []
            avg_rewards = []
            all_rewards = []
            for i in range(args["test_samples"]):
                start = time.time()
                avg_reward, rewards, top_q_distances_min, low_q_value = dec(env, agents, args["test_budget"], deploy=True, model_path=model_path, training_steps=1000000000000000000)
                end = time.time()
                # Collect Data
                testing_times.append(end-start)
                avg_rewards.append(avg_reward)
                all_rewards.extend(rewards)

            # Log Data
            avg_rewards_avg = sum(avg_rewards) / len(avg_rewards)
            testing_times_avg = sum(testing_times) / len(testing_times)
            m_logger.log("Random Testing", avg_rewards_avg, testing_times_avg)
            m_logger.store_collected_rewards("Random Testing", all_rewards, path="logs/dec_all_rewards.csv")
            del all_rewards

        # Q-Distance Testing
        print("Random Q-testing")
        q_distance_testing_times = []
        top_q_distances_mins = []
        avg_rewards = []
        all_rewards = []
        for i in range(args["test_samples"]):
            start = time.time()
            #avg_reward, top_q_distances_min = eval(env, args["model_path"], num_games = args["calibration_number"], top_percentage=args["top_percentage"])
            avg_reward, rewards, top_q_distances_min, low_q_value = dec(env, agents, args["calibration_number"], deploy=True, model_path=model_path, training_steps=1000000000000000000, top_percentage=args["top_percentage"], low_percentage=low_percentage)
            remaining_test_budget = args["test_budget"] - args["calibration_number"]
            avg_reward, rewards = dec_q_testing(env, agents, remaining_test_budget, top_q_distances_min, all_rewards=rewards)
            end = time.time()
            # Collect Data
            q_distance_testing_times.append(end-start)
            top_q_distances_mins.append(top_q_distances_min)
            avg_rewards.append(avg_reward)
            all_rewards.extend(rewards)
            #print(f"{i} Test Sample - Top-q_distance-min: {top_q_distances_min} Average reward: {avg_reward} - Time: {end-start}")
        # Log Data
        top_q_distances_mins_avg = sum(top_q_distances_mins) / len(top_q_distances_mins)
        avg_rewards_avg = sum(avg_rewards) / len(avg_rewards)
        q_distance_testing_times_avg = sum(q_distance_testing_times) / len(q_distance_testing_times)
        m_logger.log("Random Q-testing", avg_rewards_avg, q_distance_testing_times_avg)
        m_logger.store_collected_rewards("Random Q-testing", all_rewards, path="logs/dec_all_rewards.csv")
        del all_rewards

        # Q-low Testing
        print("Q-Low-Testing")
        q_distance_testing_times = []
        top_q_distances_mins = []
        avg_rewards = []
        all_rewards = []
        for i in range(args["test_samples"]):
            start = time.time()
            #avg_reward, top_q_distances_min = eval(env, args["model_path"], num_games = args["calibration_number"], top_percentage=args["top_percentage"])
            avg_reward, rewards, top_q_distances_min, low_q_value = dec(env, agents, args["calibration_number"], deploy=True, model_path=model_path, training_steps=1000000000000000000, top_percentage=args["top_percentage"], low_percentage=low_percentage)
            remaining_test_budget = args["test_budget"] - args["calibration_number"]
            avg_reward, rewards = dec_q_low_testing(env, agents, remaining_test_budget, low_q_value, all_rewards=rewards)
            end = time.time()
            # Collect Data
            q_distance_testing_times.append(end-start)
            top_q_distances_mins.append(top_q_distances_min)
            avg_rewards.append(avg_reward)
            all_rewards.extend(rewards)
            #print(f"{i} Test Sample - Top-q_distance-min: {top_q_distances_min} Average reward: {avg_reward} - Time: {end-start}")
        # Log Data
        top_q_distances_mins_avg = sum(top_q_distances_mins) / len(top_q_distances_mins)
        avg_rewards_avg = sum(avg_rewards) / len(avg_rewards)
        q_distance_testing_times_avg = sum(q_distance_testing_times) / len(q_distance_testing_times)
        m_logger.log("Random Q-low-testing", avg_rewards_avg, q_distance_testing_times_avg)
        m_logger.store_collected_rewards("Random Q-low-testing", all_rewards, path="logs/dec_all_rewards.csv")
        del all_rewards

        # Genetic Simple Spread
        print("Genetic Testing")
        
        testing_times = []
        avg_rewards = []
        all_rewards = []
        for i in range(args["test_samples"]):
            start = time.time()
            
            env, gtest = build_dec_env(args["env_name"], args["N_agents"], args["local_ratio"], True)
            
            avg_reward, rewards, top_q_distances_min, low_q_value = dec(env, agents, args["test_budget"], deploy=True, model_path=model_path, training_steps=1000000000000000000, top_percentage=args["top_percentage"], low_percentage=low_percentage, gtest=gtest)
            
            end = time.time()
            # Collect Data
            # Check if avg_reward is float
            if type(avg_reward) != float:
                print("Error: avg_reward is not a float", avg_reward, type(avg_reward))

            testing_times.append(end-start)
            avg_rewards.append(avg_reward)
            all_rewards.extend(rewards)
        # Log Data
        avg_rewards_avg = sum(avg_rewards) / len(avg_rewards)
        testing_times_avg = sum(testing_times) / len(testing_times)
        m_logger.log("Genetic Testing", avg_rewards_avg, testing_times_avg)
        m_logger.store_collected_rewards("Genetic Testing", all_rewards, path="logs/dec_all_rewards.csv")
        del all_rewards
        
        # Genetic Q-Testing
        print("Genetic Q-testing")
        testing_times = []
        avg_rewards = []
        all_rewards = []
        for i in range(args["test_samples"]):
            start = time.time()
            env, gtest = build_dec_env(args["env_name"], args["N_agents"], args["local_ratio"], False)
            avg_reward, rewards, top_q_distances_min, low_q_value = dec(env, agents, args["calibration_number"], deploy=True, model_path=model_path, training_steps=1000000000000000000, top_percentage=args["top_percentage"], low_percentage=low_percentage)
            remaining_test_budget = args["test_budget"] - args["calibration_number"]
            env, gtest = build_dec_env(args["env_name"], args["N_agents"], args["local_ratio"], True)
            avg_reward, rewards = dec_q_testing(env, agents, remaining_test_budget, top_q_distances_min, all_rewards=rewards, gtest=gtest)
            end = time.time()
            # Collect Data
            testing_times.append(end-start)
            avg_rewards.append(avg_reward)
            all_rewards.extend(rewards)
        # Log Data
        avg_rewards_avg = sum(avg_rewards) / len(avg_rewards)
        testing_times_avg = sum(testing_times) / len(testing_times)
        m_logger.log("Genetic Q-testing", avg_rewards_avg, testing_times_avg)
        m_logger.store_collected_rewards("Genetic Q-testing", all_rewards, path="logs/dec_all_rewards.csv")
        del all_rewards