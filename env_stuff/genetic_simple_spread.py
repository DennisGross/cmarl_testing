from gimitest.env_decorator import EnvDecorator
from gimitest.gtest import GTest
from pettingzoo.mpe import simple_spread_v3, simple_speaker_listener_v4
import numpy as np

def find_function_path(obj, func_name, current_path=""):
    """
    Recursively searches for a function by name in an object and returns the object-path to it.

    :param obj: The object to search within.
    :param func_name: The name of the function to search for.
    :param current_path: The current path in the object (used for recursion).
    :return: The object-path to the function, or None if not found.
    """
    # Check if the object has the function as an attribute
    if hasattr(obj, func_name) and callable(getattr(obj, func_name)):
        return f"{current_path}.{func_name}".strip(".")
    
    # Recursively search the object's attributes
    for attr_name in dir(obj):
        attr = getattr(obj, attr_name)
        # Skip built-in attributes and non-object types
        if not attr_name.startswith('__') and hasattr(attr, '__dict__'):
            result = find_function_path(attr, func_name, f"{current_path}.{attr_name}".strip("."))
            if result:
                return result

    return None

import random

import torch

import itertools
class Individual:

    def __init__(self, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3
        self.x4 = x4
        self.y4 = y4
        self.x5 = x5
        self.y5 = y5

    
    def mutate(self, mutation_rate):
        # Mutate between -1 and 1
        # Agents
        if random.random() < mutation_rate:
            self.x1 += random.uniform(-0.9, 0.9)
        if random.random() < mutation_rate:
            self.y1 += random.uniform(-0.9, 0.9)
        if random.random() < mutation_rate:
            self.x2 += random.uniform(-0.9, 0.9)
        if random.random() < mutation_rate:
            self.y2 += random.uniform(-0.9, 0.9)
        if random.random() < mutation_rate:
            self.x3 += random.uniform(-0.9, 0.9)
        if random.random() < mutation_rate:
            self.y3 += random.uniform(-0.9, 0.9)
        if random.random() < mutation_rate:
            self.x4 += random.uniform(-0.9, 0.9)
        if random.random() < mutation_rate:
            self.y4 += random.uniform(-0.9, 0.9)
        if random.random() < mutation_rate:
            self.x5 += random.uniform(-0.9, 0.9)
        if random.random() < mutation_rate:
            self.y5 += random.uniform(-0.9, 0.9)

        # Make sure the values are between -1.0 and 1.0
        self.x1 = max(min(self.x1, 0.9), -0.9)
        self.y1 = max(min(self.y1, 0.9), -0.9)
        self.x2 = max(min(self.x2, 0.9), -0.9)
        self.y2 = max(min(self.y2, 0.9), -0.9)
        self.x3 = max(min(self.x3, 0.9), -0.9)
        self.y3 = max(min(self.y3, 0.9), -0.9)
        self.x4 = max(min(self.x4, 0.9), -0.9)
        self.y4 = max(min(self.y4, 0.9), -0.9)
        self.x5 = max(min(self.x5, 0.9), -0.9)
        self.y5 = max(min(self.y5, 0.9), -0.9)

        
    def crossover(self, other, crossover_rate):
        # Crossover between two individuals
        return Individual(
            self.x1 if random.random() < crossover_rate else other.x1,
            self.y1 if random.random() < crossover_rate else other.y1,
            self.x2 if random.random() < crossover_rate else other.x2,
            self.y2 if random.random() < crossover_rate else other.y2,
            self.x3 if random.random() < crossover_rate else other.x3,
            self.y3 if random.random() < crossover_rate else other.y3,
            self.x4 if random.random() < crossover_rate else other.x4,
            self.y4 if random.random() < crossover_rate else other.y4,
            self.x5 if random.random() < crossover_rate else other.x5,
            self.y5 if random.random() < crossover_rate else other.y5,
        )


    @staticmethod
    def random_individual():
        x1 = random.uniform(-0.9, 0.9)
        y1 = random.uniform(-0.9, 0.9)
        x2 = random.uniform(-0.9, 0.9)
        y2 = random.uniform(-0.9, 0.9)
        x3 = random.uniform(-0.9, 0.9)
        y3 = random.uniform(-0.9, 0.9)
        x4 = random.uniform(-0.9, 0.9)
        y4 = random.uniform(-0.9, 0.9)
        x5 = random.uniform(-0.9, 0.9)
        y5 = random.uniform(-0.9, 0.9)
        return Individual(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5)

    def __str__(self):
        return f"Individual(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}, x3={self.x3}, y3={self.y3}, x4={self.x4}, y4={self.y4}, x5={self.x5}, y5={self.y5})"
    
class GGTest(GTest):

    def __init__(self, env, population_size, N_agents):
        super().__init__(env)
        # Add code to env.reset_world(self, world, np_random)
        self.decorate_reset_world()
        # Genetic Algorithm parameters
        self.population_size = population_size
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.init_step = True
        self.individual_index = 0
        self.population = self.populate([], self.population_size)
        self.N_agents = N_agents
        
        
    def populate(self, population, population_size):
        for i in range(population_size): 
            population.append((Individual.random_individual(), None))
        return population


    def get_ith_best_individual(self, population, i):
        return sorted(population, key=lambda x: x[1], reverse=False)[i]

    def all_fitnesses_calculated(self, population):
        for individual, fitness in population:
            if fitness is None:
                return False
        return True

    def post_reset_world(self):
        if self.individual_index>=self.population_size:
            best_individual = self.get_ith_best_individual(self.population, 0)[0]
            second_best_individual = self.get_ith_best_individual(self.population, 1)[0]
            self.population = []
            # Repopulate with mutated children
            for i in range(self.population_size): 
                child = best_individual.crossover(second_best_individual, self.crossover_rate)
                child.mutate(self.mutation_rate)
                self.population.append((child, None))
            self.individual_index = 0
        
        # Get Individual
        next_individual, fitness = self.population[self.individual_index]
        # Update State
        if self.N_agents == 2:
            self.env.unwrapped.world.agents[0].state.p_pos = np.array([next_individual.x1, next_individual.y1])
            self.env.unwrapped.world.agents[1].state.p_pos = np.array([next_individual.x2, next_individual.y2])
        elif self.N_agents == 3:
            self.env.unwrapped.world.agents[0].state.p_pos = np.array([next_individual.x1, next_individual.y1])
            self.env.unwrapped.world.agents[1].state.p_pos = np.array([next_individual.x2, next_individual.y2])
            self.env.unwrapped.world.agents[2].state.p_pos = np.array([next_individual.x3, next_individual.y3])
        elif self.N_agents == 4:
            self.env.unwrapped.world.agents[0].state.p_pos = np.array([next_individual.x1, next_individual.y1])
            self.env.unwrapped.world.agents[1].state.p_pos = np.array([next_individual.x2, next_individual.y2])
            self.env.unwrapped.world.agents[2].state.p_pos = np.array([next_individual.x3, next_individual.y3])
            self.env.unwrapped.world.agents[3].state.p_pos = np.array([next_individual.x4, next_individual.y4])
        elif self.N_agents == 5:
            self.env.unwrapped.world.agents[0].state.p_pos = np.array([next_individual.x1, next_individual.y1])
            self.env.unwrapped.world.agents[1].state.p_pos = np.array([next_individual.x2, next_individual.y2])
            self.env.unwrapped.world.agents[2].state.p_pos = np.array([next_individual.x3, next_individual.y3])
            self.env.unwrapped.world.agents[3].state.p_pos = np.array([next_individual.x4, next_individual.y4])
            self.env.unwrapped.world.agents[4].state.p_pos = np.array([next_individual.x5, next_individual.y5])


    def set_individual_fitness(self, fitness):
        self.population[self.individual_index] = (self.population[self.individual_index][0], fitness)
        self.individual_index += 1
       
    
    def decorate_reset_world(self):
        # Decorating env.reset_world to include custom behavior
        original_reset_world = self.env.aec_env.env.env.scenario.reset_world
        post_reset_function =  self.post_reset_world

        
        def custom_reset_world(*args, **kwargs):
            # Call the original reset_world function
            result = original_reset_world(*args, **kwargs)
            
            # Post-reset_world behavior
            post_reset_function()

            return result
        
        # Replace the original reset_world with the decorated one
        self.env.aec_env.env.env.scenario.reset_world = custom_reset_world

if __name__ == "__main__":
    env = simple_speaker_listener_v4.parallel_env()
    print(find_function_path(env, "reset_world"))
    #exit(0)
    gtest = GGTest(env, 5)
    for i in range(100):
        env.reset()
        gtest.set_individual_fitness(0.5)
