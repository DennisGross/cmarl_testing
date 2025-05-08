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



def sum_distances(scalars):
    return sum(abs(a - b) for a, b in itertools.combinations(scalars, 2))



def train(
    env, save_model_path, neurons=[512, 512, 512], learning_rate=1e-3, training_steps = 10_000, seed = 0, num_cpus = 20, num_instances = 256
):
    
    env.reset(seed=seed)

    #print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_instances, num_cpus=num_cpus, base_class="stable_baselines3")

    # Define the policy with a custom neural network architecture
    policy_kwargs = dict(
        net_arch=neurons,  # Two hidden layers with 256 and 128 neurons
    )

    # Create the DQN model with the custom architecture
    model = DQN(
        MlpPolicy,
        env,
        verbose=3,
        learning_rate=learning_rate,
        batch_size=256,
        policy_kwargs=policy_kwargs
    )

    model.learn(total_timesteps=training_steps)

    model.save(save_model_path)

    #print("Model has been saved.")

    #print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()
    


def calibrate(env, model_path, num_games = 100, top_percentage=0.9, low_percentage=0.1, gtest=None):
    # Evaluate a trained agent vs a random agent

    try:
        latest_policy = max(
            glob.glob(model_path), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = DQN.load(latest_policy)  # Load DQN
    
    # Evaluate the trained model
    rewards = []
    all_q_distances = []
    all_max_q_values = []
    for i in range(num_games):
        obs, _ = env.reset()
        max_q_values = []
        for agent in env.agents:
            obs_tensor = torch.tensor([obs[agent]], dtype=torch.float32).to(model.device)
            q_values = model.q_net(obs_tensor)
            #print(q_values)
            max_q_values.append(torch.max(q_values).item())
        all_q_distances.append(sum_distances(max_q_values))
        all_max_q_values.extend(max_q_values)

        done = False
        episode_reward = 0
        while not done:
            actions = {}
            
            for agent in env.agents:
                actions[agent] = model.predict(obs[agent], deterministic=True)[0]
                
            obs, reward, termination, truncation, info = env.step(actions)
       
            # Sum the rewards that each agent received
            episode_reward += sum(reward.values())
            done = all(termination.values()) and all(truncation.values())

        if math.isnan(episode_reward) == False:
            rewards.append(episode_reward)
            if gtest:
                gtest.set_individual_fitness(episode_reward)
        else:
            if gtest:
                gtest.set_individual_fitness(math.inf)

    
    avg_reward = sum(rewards) / len(rewards)
    

    # Get the top 10% quantile of q_distances
    top_q_distance = np.quantile(all_q_distances, top_percentage)

    low_q_value = np.quantile(all_max_q_values, low_percentage)
    
    
    return avg_reward, rewards, top_q_distance, low_q_value

def q_distance_testing(env, model_path, num_games = 100, top_q_distance=100, rewards=[], gtest=None):
    # Evaluate a trained agent vs a random agent
    try:
        latest_policy = max(
            glob.glob(model_path), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = DQN.load(latest_policy)  # Load DQN
    
    # Evaluate the trained model
  

    episode = 0
    while episode < num_games:
        obs, _ = env.reset()
        max_q_values = []
        q_values = {}
        print("====================================")
        for agent in env.agents:
            obs_tensor = torch.tensor([obs[agent]], dtype=torch.float32).to(model.device)
            q_values = model.q_net(obs_tensor)
            print(q_values)
            max_q_values.append(torch.max(q_values).item())
        if sum_distances(max_q_values)<top_q_distance:
            if gtest:
                gtest.set_individual_fitness(math.inf)
            continue
        print(sum_distances(max_q_values), top_q_distance)
        

        done = False
        episode_reward = 0
        while not done:
            actions = {}
            for agent in env.agents:
                actions[agent] = model.predict(obs[agent], deterministic=True)[0]
                
            obs, reward, termination, truncation, info = env.step(actions)
       
            # Sum the rewards that each agent received
            episode_reward += sum(reward.values())
            done = all(termination.values()) and all(truncation.values())
        
        print(episode_reward)
        if math.isnan(episode_reward) == False:
            rewards.append(episode_reward)
            if gtest:
                gtest.set_individual_fitness(episode_reward)
        else:
            if gtest:
                gtest.set_individual_fitness(math.inf)
        
        episode += 1


    avg_reward = sum(rewards) / len(rewards)
    
    
    return avg_reward, rewards


def low_q_testing(env, model_path, q_value, num_games = 100, rewards=[], gtest=None):
    # Evaluate a trained agent vs a random agent
    try:
        latest_policy = max(
            glob.glob(model_path), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = DQN.load(latest_policy)  # Load DQN
    
    # Evaluate the trained model
  

    episode = 0
    while episode < num_games:
        obs, _ = env.reset()
        max_q_values = []
        for agent in env.agents:
            obs_tensor = torch.tensor([obs[agent]], dtype=torch.float32).to(model.device)
            q_values = model.q_net(obs_tensor)
            max_q_values.append(torch.max(q_values).item())
        # Check if all max_q_values are below q_value
        if all(q_value > x for x in max_q_values):
            pass
        else:
            continue

        done = False
        episode_reward = 0
        while not done:
            actions = {}
            for agent in env.agents:
                actions[agent] = model.predict(obs[agent], deterministic=True)[0]
                
            obs, reward, termination, truncation, info = env.step(actions)
       
            # Sum the rewards that each agent received
            episode_reward += sum(reward.values())
            done = all(termination.values()) and all(truncation.values())
        if math.isnan(episode_reward) == False:
            rewards.append(episode_reward)
            if gtest:
                gtest.set_individual_fitness(episode_reward)
        else:
            if gtest:
                gtest.set_individual_fitness(math.inf)
        
        episode += 1


    avg_reward = sum(rewards) / len(rewards)
    
    
    return avg_reward, rewards



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

def build_env(env_name, N_agents, local_ratio, is_gtest=False):
    gtest = None
    if env_name == "simple_speaker_listener_v4":
        env = simple_speaker_listener_v4.parallel_env()#, render_mode="human")
        if is_gtest:
            gtest = env_stuff.genetic_simple_spread.GGTest(env, 200, N_agents=N_agents)
        env = ss.black_death_v3(env)
        env = ss.multiagent_wrappers.pad_observations_v0(env)
        env = ss.multiagent_wrappers.pad_action_space_v0(env)
    elif env_name == "simple_spread_v3":
        env = simple_spread_v3.parallel_env(local_ratio=local_ratio, N=N_agents)#, render_mode="human")
        if is_gtest:
            gtest = env_stuff.genetic_simple_spread.GGTest(env, 200, N_agents=N_agents)
    elif env_name == "knight_wizard_zombies":
        env = env_stuff.knight_wizard_zombies.KnightWizardZombies()
        if is_gtest:
            gtest = env_stuff.genetic_knight_wizard_zombies.GGTest(env, 200)
    elif env_name == "coin_collector":
        env = env_stuff.coin_collector.CoinGame()
        if is_gtest:
            gtest = env_stuff.genetic_coin_collector.GGTest(env, 200)
        
    return env, gtest


if __name__ == "__main__":
    args = get_args()
    low_percentage = 0.75
    print(args)
    env, _ = build_env(args["env_name"], args["N_agents"], args["local_ratio"])
    if args["train"]:
        print("Training")
        train(env=env, save_model_path=args["model_path"], training_steps=args["training_steps"], num_cpus=args["num_cpus"], num_instances=args["num_instances"])
    else:
        print("Testing")
        m_logger = Logger("logs/central_logs.csv", args["env_name"],args["N_agents"],args["local_ratio"],args["model_path"], args["calibration_number"], args["test_budget"], args["top_percentage"])

        if True:
            # Random Testing
            print("Random Testing")
            testing_times = []
            avg_rewards = []
            all_rewards = []
            for i in range(args["test_samples"]):
                start = time.time()
                avg_reward, rewards, top_q_distances_min, low_q_value = calibrate(env, args["model_path"],  num_games = args["test_budget"], top_percentage=args["top_percentage"], low_percentage=low_percentage)
                end = time.time()
                # Collect Data
                testing_times.append(end-start)
                avg_rewards.append(avg_reward)
                all_rewards.extend(rewards)

            # Log Data
            avg_rewards_avg = sum(avg_rewards) / len(avg_rewards)
            testing_times_avg = sum(testing_times) / len(testing_times)
            m_logger.log("Random Testing", avg_rewards_avg, testing_times_avg)
            m_logger.store_collected_rewards("Random Testing", all_rewards, path="logs/central_all_rewards.csv")
            del all_rewards       
            
        # Q-testing Testing
        print("Q-testing")
        q_distance_testing_times = []
        top_q_distances_mins = []
        avg_rewards = []
        all_rewards = []
        for i in range(args["test_samples"]):
            start = time.time()
            #avg_reward, top_q_distances_min = eval(env, args["model_path"], num_games = args["calibration_number"], top_percentage=args["top_percentage"])
            avg_reward, rewards, top_q_distances_min, low_q_value  = calibrate(env, args["model_path"],  num_games = args["calibration_number"], top_percentage=args["top_percentage"], low_percentage=low_percentage)
            remaining_test_budget = args["test_budget"] - args["calibration_number"]
            avg_reward, rewards = q_distance_testing(env, args["model_path"], remaining_test_budget, top_q_distances_min, rewards=rewards)
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
        m_logger.store_collected_rewards("Random Q-testing", all_rewards, path="logs/central_all_rewards.csv")
        del all_rewards

        # Q-low Testing
        if False:
            print("Q-low Testing")
            q_distance_testing_times = []
            top_q_distances_mins = []
            avg_rewards = []
            all_rewards = []
            for i in range(args["test_samples"]):
                start = time.time()
                #avg_reward, top_q_distances_min = eval(env, args["model_path"], num_games = args["calibration_number"], top_percentage=args["top_percentage"])
                avg_reward, rewards, top_q_distances_min, low_q_value  = calibrate(env, args["model_path"],  num_games = args["calibration_number"], top_percentage=args["top_percentage"], low_percentage=low_percentage)
                remaining_test_budget = args["test_budget"] - args["calibration_number"]
                avg_reward, rewards = low_q_testing(env, args["model_path"], remaining_test_budget, top_q_distances_min, rewards=rewards)
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
            m_logger.store_collected_rewards("Random Q-low-testing", all_rewards, path="logs/central_all_rewards.csv")
            del all_rewards
        
         

        # Genetic Simple Spread
        print("Genetic Testing")
        
        testing_times = []
        avg_rewards = []
        all_rewards = []
        for i in range(args["test_samples"]):
            start = time.time()
            
            env, gtest = build_env(args["env_name"], args["N_agents"], args["local_ratio"], True)
            
            avg_reward, rewards, top_q_distances_min, low_q_value = calibrate(env, args["model_path"],  num_games = args["test_budget"], top_percentage=args["top_percentage"], low_percentage=low_percentage, gtest=gtest)
            
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
        m_logger.store_collected_rewards("Genetic Testing", all_rewards, path="logs/central_all_rewards.csv")
        del all_rewards

        # Genetic Q-Testing
        print("Genetic Q-testing")
        testing_times = []
        avg_rewards = []
        all_rewards = []
        for i in range(args["test_samples"]):
            start = time.time()
            avg_reward, rewards, top_q_distances_min, low_q_value  = calibrate(env, args["model_path"],  num_games = args["calibration_number"], top_percentage=args["top_percentage"], low_percentage=low_percentage)
            remaining_test_budget = args["test_budget"] - args["calibration_number"]
            env, gtest = build_env(args["env_name"], args["N_agents"], args["local_ratio"], True)
            avg_reward, rewards = q_distance_testing(env, args["model_path"], remaining_test_budget, top_q_distances_min, rewards=rewards, gtest=gtest)
            end = time.time()
            # Collect Data
            testing_times.append(end-start)
            avg_rewards.append(avg_reward)
            all_rewards.extend(rewards)
        # Log Data
        avg_rewards_avg = sum(avg_rewards) / len(avg_rewards)
        testing_times_avg = sum(testing_times) / len(testing_times)
        m_logger.log("Genetic Q-testing", avg_rewards_avg, testing_times_avg)
        m_logger.store_collected_rewards("Genetic Q-testing", all_rewards, path="logs/central_all_rewards.csv")
        del all_rewards
        

        
        del m_logger

    # Close Environment
    env.close()
    # Clean Up
    del env
    # GPU Memory
    torch.cuda.empty_cache()
    # Clean up stablebaseline3
    gc.collect()
    