from pettingzoo import ParallelEnv
import random
import pygame
import sys
from gymnasium import spaces
import gymnasium
import time
import numpy as np
class KnightWizardZombies(ParallelEnv):
    metadata = {
        "name": "knight_wizard_zombies",
    }

    def __init__(self):
        self.ZOMBIE_ACCURACY = 0.5
        self.MAX_X = 6
        self.MAX_Y = 7
        self.MAX_HP = 3
        self.MAX_MP = 4
        self.MAX_STEPS = 100
        self.current_step = 0
        self.knight_x = None
        self.knight_y = None
        self.knight_hp = self.MAX_HP
        self.wizard_x = None
        self.wizard_y = None
        self.wizard_hp = self.MAX_HP
        self.wizard_mp = self.MAX_MP
        self.zombie1_x = None
        self.zombie1_y = None
        self.zombie2_x = None
        self.zombie2_y = None
        self.zombie3_x = None
        self.zombie3_y = None
        
        self.agents = ["knight", "wizard"]
        self.possible_agents = ["knight", "wizard"]
        self.render_mode = ""

        self.observation_spaces = {
        "knight": spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32),
        "wizard": spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32),
        }

        self.action_spaces = {
            "knight": gymnasium.spaces.discrete.Discrete(5),
            "wizard": gymnasium.spaces.discrete.Discrete(5),
        }

    def state(self):
        return (
            self.knight_x,
            self.knight_y,
            self.knight_hp,
            self.wizard_x,
            self.wizard_y,
            self.wizard_hp,
            self.wizard_mp,
            self.zombie1_x,
            self.zombie1_y,
            self.zombie2_x,
            self.zombie2_y,
            self.zombie3_x,
            self.zombie3_y,
        )

    def reset(self, seed=None, options=None):
        self.knight_x = random.randint(0, self.MAX_X-1)
        self.knight_y = random.randint(0, self.MAX_Y-2)
        self.knight_hp = self.MAX_HP
        self.wizard_x = random.randint(0, self.MAX_X-1)
        self.wizard_y = random.randint(0, self.MAX_Y-2)
        self.wizard_hp = self.MAX_HP
        self.wizard_mp = self.MAX_MP
        self.zombie1_x = random.randint(0, self.MAX_X-1)
        self.zombie1_y = self.MAX_Y-1
        self.zombie2_x = random.randint(0, self.MAX_X-1)
        self.zombie2_y = self.MAX_Y-1
        self.zombie3_x = random.randint(0, self.MAX_X-1)
        self.zombie3_y = self.MAX_Y-1
        self.current_step = 0
        

        

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return self.get_observations(), infos


    def get_observations(self):
        observations = {
            "knight": (
                0,
                self.knight_x,
                self.knight_y,
                self.wizard_x,
                self.wizard_y,
                self.knight_hp,
                self.wizard_hp,
                self.wizard_mp,
                self.zombie1_x,
                self.zombie1_y,
                self.zombie2_x,
                self.zombie2_y,
                self.zombie3_x,
                self.zombie3_y,
            ),
            "wizard": (
                1,
                self.wizard_x,
                self.wizard_y,
                self.knight_x,
                self.knight_y,
                self.knight_hp,
                self.wizard_hp,
                self.wizard_mp,
                self.zombie1_x,
                self.zombie1_y,
                self.zombie2_x,
                self.zombie2_y,
                self.zombie3_x,
                self.zombie3_y,
            ),
        }
        return observations


    def step(self, actions):
        done = False
        rewards = {a: -2 for a in self.agents}
        healed = False

        knight_action = actions["knight"]
        wizard_action = actions["wizard"]

        zombie1_dead = False
        zombie2_dead = False
        zombie3_dead = False

        # Knight actions
        if knight_action == 0:
            self.knight_x += 1
            if self.knight_x >= self.MAX_X:
                self.knight_x = self.MAX_X-1
        elif knight_action == 1:
            self.knight_x -= 1
            if self.knight_x < 0:
                self.knight_x = 0
        elif knight_action == 2:
            self.knight_y += 1
            if self.knight_y >= self.MAX_Y:
                self.knight_y = self.MAX_Y-1
        elif knight_action == 3:
            self.knight_y -= 1
            if self.knight_y < 0:
                self.knight_y = 0
        elif knight_action == 4:
            # ATTACK
            if self.knight_x == self.zombie1_x and self.knight_y == self.zombie1_y:
                zombie1_dead = True
                
            elif self.knight_x == self.zombie2_x and self.knight_y == self.zombie2_y:
                zombie2_dead = True
            elif self.knight_x == self.zombie3_x and self.knight_y == self.zombie3_y:
                zombie3_dead = True
    

        # Wizard actions
        if wizard_action == 0:
            self.wizard_x += 1
            if self.wizard_x >= self.MAX_X:
                self.wizard_x = self.MAX_X-1
        elif wizard_action == 1:
            self.wizard_x -= 1
            if self.wizard_x < 0:
                self.wizard_x = 0
        elif wizard_action == 2:
            self.wizard_y += 1
            if self.wizard_y >= self.MAX_Y:
                self.wizard_y = self.MAX_Y-1
        elif wizard_action == 3:
            self.wizard_y -= 1
            if self.wizard_y < 0:
                self.wizard_y = 0
        elif wizard_action == 4:
            if self.wizard_mp >= 3:
                self.wizard_mp -= 3
                self.knight_hp += 2
                healed = True

        # Zombie attacks
        if self.zombie1_x == self.knight_x and self.zombie1_y == self.knight_y and random.random() <= self.ZOMBIE_ACCURACY and zombie1_dead == False:
            self.knight_hp -= 1
        if self.zombie2_x == self.knight_x and self.zombie2_y == self.knight_y and random.random() <= self.ZOMBIE_ACCURACY and zombie2_dead == False:
            self.knight_hp -= 1
        if self.zombie3_x == self.knight_x and self.zombie3_y == self.knight_y and random.random() <= self.ZOMBIE_ACCURACY and zombie3_dead == False:
            self.knight_hp -= 1
        if self.zombie1_x == self.wizard_x and self.zombie1_y == self.wizard_y and random.random() <= self.ZOMBIE_ACCURACY and zombie1_dead == False:
            self.wizard_hp -= 1
        if self.zombie2_x == self.wizard_x and self.zombie2_y == self.wizard_y and random.random() <= self.ZOMBIE_ACCURACY and zombie2_dead == False:
            self.wizard_hp -= 1
        if self.zombie3_x == self.wizard_x and self.zombie3_y == self.wizard_y and random.random() <= self.ZOMBIE_ACCURACY and zombie3_dead == False:
            self.wizard_hp -= 1

        # Reset dead zombies
        if zombie1_dead:
            self.zombie1_x = 0
            self.zombie1_y = self.MAX_Y-1
        if zombie2_dead:
            self.zombie2_x = 2
            self.zombie2_y = self.MAX_Y-1
        if zombie3_dead:
            self.zombie3_x = 4
            self.zombie3_y = self.MAX_Y-1
        
        # Move zombie1
        random_action = random.randint(0, 2)
        if random_action == 0 and zombie1_dead == False:
            self.zombie1_x += 1
            if self.zombie1_x >= self.MAX_X:
                self.zombie1_x = self.MAX_X-1
        elif random_action == 1 and zombie1_dead == False:
            self.zombie1_x -= 1
            if self.zombie1_x < 0:
                self.zombie1_x = 0
        elif random_action == 2 and zombie1_dead == False:
            self.zombie1_y -= 1
            if self.zombie1_y < 0:
                self.zombie1_y = 0
        
        # Move zombie2
        random_action = random.randint(0, 2)
        if random_action == 0 and zombie2_dead == False:
            self.zombie2_x += 1
            if self.zombie2_x >= self.MAX_X:
                self.zombie2_x = self.MAX_X-1
        elif random_action == 1 and zombie2_dead == False:
            self.zombie2_x -= 1
            if self.zombie2_x < 0:
                self.zombie2_x = 0
        elif random_action == 2 and zombie2_dead == False:
            self.zombie2_y -= 1
            if self.zombie2_y < 0:
                self.zombie2_y = 0

        # Move zombie3
        random_action = random.randint(0, 2)
        if random_action == 0 and zombie3_dead == False:
            self.zombie3_x += 1
            if self.zombie3_x >= self.MAX_X:
                self.zombie3_x = self.MAX_X-1
        elif random_action == 1 and zombie3_dead == False:
            self.zombie3_x -= 1
            if self.zombie3_x < 0:
                self.zombie3_x = 0
        elif random_action == 2 and zombie3_dead == False:
            self.zombie3_y -= 1
            if self.zombie3_y < 0:
                self.zombie3_y = 0

        # Increase wizard mp
        self.wizard_mp += 1

        # Check if game is over
        if self.knight_hp <= 0 or self.wizard_hp <= 0 or self.zombie1_y <= 0 or self.zombie2_y <= 0 or self.zombie3_y <= 0:
            done = True
            rewards["wizard"] -= 200
            rewards["knight"] -= 200

        if healed:
            rewards["wizard"] += 1
            rewards["knight"] += 1
        if zombie1_dead or zombie2_dead or zombie3_dead:
            rewards["knight"] += 1
            rewards["wizard"] += 1

        self.current_step += 1
        if self.current_step >= self.MAX_STEPS:
            done = True
       
        infos = {a: {} for a in self.agents}

        # dones
        done = {a: done for a in self.agents}

        return self.get_observations(), rewards, done, done, infos

    def render(self):
        # Initialize pygame if not already initialized
        if not pygame.get_init():
            pygame.init()
                
        # Define some constants
        TILE_SIZE = 60
        WIDTH = self.MAX_X * TILE_SIZE  # Width is aligned with X dimension
        HEIGHT = self.MAX_Y * TILE_SIZE  # Height is aligned with Y dimension

        # Define colors
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)
        RED = (255, 0, 0)

        # Set up the display
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Knight, Wizard, and Zombies")

        # Create a clock object to control the frame rate
        clock = pygame.time.Clock()

        # Main render loop
        screen.fill(WHITE)

        # Draw the grid
        for x in range(self.MAX_X):
            for y in range(self.MAX_Y):
                pygame.draw.rect(screen, BLACK, pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE), 1)

        # Draw the knight
        if self.knight_hp > 0:
            knight_rect = pygame.Rect(self.knight_x * TILE_SIZE, self.knight_y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, BLUE, knight_rect)

        # Draw the wizard
        if self.wizard_hp > 0:
            wizard_rect = pygame.Rect(self.wizard_x * TILE_SIZE, self.wizard_y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, GREEN, wizard_rect)

        # Draw the zombies
        zombie_color = RED
        if self.zombie1_x is not None and self.zombie1_y is not None:
            zombie1_rect = pygame.Rect(self.zombie1_x * TILE_SIZE, self.zombie1_y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, zombie_color, zombie1_rect)

        if self.zombie2_x is not None and self.zombie2_y is not None:
            zombie2_rect = pygame.Rect(self.zombie2_x * TILE_SIZE, self.zombie2_y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, zombie_color, zombie2_rect)

        if self.zombie3_x is not None and self.zombie3_y is not None:
            zombie3_rect = pygame.Rect(self.zombie3_x * TILE_SIZE, self.zombie3_y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, zombie_color, zombie3_rect)

        # Update the display
        pygame.display.flip()

        # Handle events (like closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Control the frame rate
        clock.tick(30)
        

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]


if __name__ == "__main__":
    env = KnightWizardZombies()
    env.reset()
    env.render()
    done = False
    while not done:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        observations, rewards, done, truncat, infos = env.step(actions)
        env.render()
        done = done or truncat
        print(observations)
        print(rewards)
        print(done)
        print(infos)
        print()
        time.sleep(0.5)