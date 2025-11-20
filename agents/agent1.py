from agents.base_agent import Agent
from environment.actions import Action
import numpy as np
import random
import os

class MarcAgent(Agent):
    def __init__(self, x, y, role, grid_size = 20):
        super().__init__(x, y, role)
        self.alpha = 0.9
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.min_epsilon = 0.05

        self.num_states = grid_size * grid_size
        self.num_action = len(Action)

        self.q_table = np.zeros((self.num_states, self.num_action))

        self.last_state = None
        self.last_action = None

        self.visited = set()

    def decide_move(self, percepts, grid_size):
        state = self.get_state_id()

        if random.random() < self.epsilon:
            action_id = random.randint(0, self.num_action - 1)
        else:
            action_id = np.argmax(self.q_table[state])

        action = self.id_to_action(action_id)

        self.last_state = state
        self.last_action = action_id

        return action
    

    def learn(self, reward, next_state):
        old_value = self.q_table[self.last_state, self.last_action]
        next_max = np.max(self.q_table[next_state])

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[self.last_state, self.last_action] = new_value

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def reward_from_result(self, result, percepts):
        if result == "WIN":
            return 100
        if result == "DIED":
            return -100
        
        return -1


    def get_state_id(self):
        x, y = self.pos()
        return y * 20 + x
    
    def save_q_table(self, filename="q_table_marcagent.npy"):
        """Speichere Q-Table in Datei"""
        np.save(filename, self.q_table)
    
    def load_q_table(self, filename="q_table_marcagent.npy"):
        """Lade Q-Table aus Datei, falls vorhanden"""
        if os.path.exists(filename):
            self.q_table = np.load(filename)
            return True
        return False

    ACTIONS = [
        Action.MOVE_UP,
        Action.MOVE_DOWN,
        Action.MOVE_LEFT,
        Action.MOVE_RIGHT,
        Action.GRAB,
        Action.SHOOT_UP,
        Action.SHOOT_DOWN,
        Action.SHOOT_LEFT,
        Action.SHOOT_RIGHT
    ]

    def action_to_id(self, action):
        return self.ACTIONS.index(action)
    
    def id_to_action(self, idx):
        return self.ACTIONS[idx]