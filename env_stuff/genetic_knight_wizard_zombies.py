from gimitest.env_decorator import EnvDecorator
from gimitest.gtest import GTest
from env_stuff.knight_wizard_zombies import *
import numpy as np
import math

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

    def __init__(self, x1, y1, x2, y2):
        self.MAX_X = 6
        self.MAX_Y = 7
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    
    def mutate(self, mutation_rate):
        # Mutate between -1 and 1
        # Agents
        if random.random() < mutation_rate:
            self.x1 += random.randint(-self.MAX_X+1, self.MAX_X-1)
        if random.random() < mutation_rate:
            self.y1 += random.randint(-self.MAX_Y+1, self.MAX_Y-1)
        if random.random() < mutation_rate:
            self.x2 += random.randint(-self.MAX_X+1, self.MAX_X-1)
        if random.random() < mutation_rate:
            self.y2 += random.randint(-self.MAX_Y+1, self.MAX_Y-1)
        

        # Make sure the values are between -1.0 and 1.0
        self.x1 = max(min(self.x1, self.MAX_X-1), 0)
        self.y1 = max(min(self.y1, self.MAX_Y-1), 0)
        self.x2 = max(min(self.x2, self.MAX_X-1), 0)
        self.y2 = max(min(self.y2, self.MAX_Y-1), 0)
        

        
    def crossover(self, other, crossover_rate):
        # Crossover between two individuals
        return Individual(
            self.x1 if random.random() < crossover_rate else other.x1,
            self.y1 if random.random() < crossover_rate else other.y1,
            self.x2 if random.random() < crossover_rate else other.x2,
            self.y2 if random.random() < crossover_rate else other.y2,
        )


    @staticmethod
    def random_individual():
        indi = Individual(random.randint(-6, 6), random.randint(-7, 7), random.randint(-6, 6), random.randint(-7, 7))
        x1 = random.randint(0, indi.MAX_X-1)
        y1 = random.randint(0, indi.MAX_Y-1)
        x2 = random.randint(0, indi.MAX_X-1)
        y2 = random.randint(0, indi.MAX_Y-1)
        return Individual(x1, y1, x2, y2)

    def __str__(self):
        return f"Individual(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}, x3={self.x3}, y3={self.y3}, x4={self.x4}, y4={self.y4}, x5={self.x5}, y5={self.y5})"
    
class GGTest(GTest):

    def __init__(self, env, population_size):
        super().__init__(env)
        # Add code to env.reset_world(self, world, np_random)
        # Genetic Algorithm parameters
        self.population_size = population_size
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.init_step = True
        self.individual_index = 0
        self.population = self.populate([], self.population_size)
        
        
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

    def post_reset_configuration(self, next_state):
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
        self.env.unwrapped.knight_x = next_individual.x1
        self.env.unwrapped.knight_y = next_individual.y1
        self.env.unwrapped.wizard_x = next_individual.x2
        self.env.unwrapped.wizard_y = next_individual.y2
        return self.env.unwrapped.get_observations(), {a: {} for a in self.env.unwrapped.possible_agents}


    def set_individual_fitness(self, fitness):
        if self.individual_index >= self.population_size:
            self.individual_index = 0
        self.population[self.individual_index] = (self.population[self.individual_index][0], fitness)
        self.individual_index += 1
       
    
    

if __name__ == "__main__":
    env = KnightWizardZombies()
    #exit(0)
    gtest = GGTest(env, 5)
    EnvDecorator.decorate(env, gtest)
    for i in range(100):
        env.reset()
        gtest.set_individual_fitness(0.5)
