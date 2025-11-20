from agents.base_agent import Agent
from environment.actions import Action
import numpy as np
import random
import os

class MarcAgent(Agent):
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

    def __init__(self, x, y, role, grid_size=20):
        super().__init__(x, y, role)

        # RL-Parameter
        self.alpha = 0.3
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.min_epsilon = 0.1

        self.grid_size = grid_size

        # Q als Dictionary: key = (state, action), value = Q-Wert
        self.q = {}

        # Speicher für letzte (s, a)
        self.last_state = None
        self.last_action = None

        # Knowledge
        self.safe = set()
        self.unsafe = set()
        self.visited = set()
        self.frontier = set()

    #  Q-Funktionen 
    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)
    
    def set_q(self, state, action, value):
        self.q[(state, action)] = value

    #  Entscheidungslogik 
    def decide_move(self, percepts, grid_size):
        # Percepts in Knowledge übernehmen
        self.update_knowledge(percepts)

        # State aus Knowledge bauen
        state = self.get_state_id(percepts)

        # Wenn Gold sichtbar, sofort greifen
        if percepts.get('glitter', False):
            return Action.GRAB

        # Exploration vs. Exploitation
        if random.random() < self.epsilon:
            # erkundet mit Knowledge
            action = self.pick_action_with_knowledge()
        else:
            # beste Aktion laut Q
            qs = [self.get_q(state, a) for a in self.ACTIONS]
            max_q = max(qs)
            # falls mehrere gleich gut sind → zufällig einen nehmen
            best_indices = [i for i, v in enumerate(qs) if v == max_q]
            action = self.ACTIONS[random.choice(best_indices)]

        # für Learning merken
        self.last_state = state
        self.last_action = action

        return action

    #  Lernen 
    def learn(self, reward, next_percepts=None):
        next_state = self.get_state_id(next_percepts)

        if self.last_state is None or self.last_action is None:
            return  # Schutz vor zu frühem learn-Aufruf

        old_q = self.get_q(self.last_state, self.last_action)
        next_max = max(self.get_q(next_state, a) for a in self.ACTIONS)

        new_q = (1 - self.alpha) * old_q + self.alpha * (reward + self.gamma * next_max)
        self.set_q(self.last_state, self.last_action, new_q)

        # Epsilon-Decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    #  Reward 
    def reward_from_result(self, result, percepts):
        if result == "WIN":
            return 100
        if result == "DIED":
            return -100
        # Bonus, wenn Agent eine Glitter-Wahrnehmung hatte (nähert sich Gold)
        if percepts.get('glitter', False):
            return 10

        # hier später shaping mit Percepts einbauen
        return -1

    #  State-Encoding aus Knowledge 
    def get_state_id(self, percepts):
        b = 1 if percepts["breeze"] else 0
        s = 1 if percepts["stench"] else 0
        g = 1 if percepts["glitter"] else 0
        return (b << 2) | (s << 1) | g

    #  Q speichern / laden (Dictionary) 
    def save_q_table(self, filename="q_table_marcagent.npy"):
        """Speichere Q-Dict in Datei"""
        np.save(filename, self.q, allow_pickle=True)
    
    def load_q_table(self, filename="q_table_marcagent.npy"):
        """Lade Q-Dict aus Datei, falls vorhanden"""
        if os.path.exists(filename):
            self.q = np.load(filename, allow_pickle=True).item()
            return True
        return False

    
    def action_to_id(self, action):
        return self.ACTIONS.index(action)
    
    def id_to_action(self, idx):
        return self.ACTIONS[idx]

    #  Episode Reset (Knowledge löschen)
    def reset_episode(self):
        """Lösche Knowledge und Lernstate für neue Episode"""
        self.safe.clear()
        self.unsafe.clear()
        self.visited.clear()
        self.frontier.clear()
        self.last_state = None
        self.last_action = None

    #  Knowledge-Update 
    def update_knowledge(self, percepts):
        x, y = self.pos()
        pos = (x, y)

        self.visited.add(pos)
        self.safe.add(pos)

        neighbors = [
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1)
        ]
        neighbors = [
            (nx, ny)
            for nx, ny in neighbors
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size
        ]

        # Keine Gefahr wahrgenommen → Nachbarn sind safe
        if not percepts.get('stench', False) and not percepts.get('breeze', False):
            for n in neighbors:
                if n not in self.visited:
                    self.safe.add(n)
                    self.frontier.add(n)
            return

        # Gefahr wahrgenommen → Nachbarn potenziell unsicher
        for n in neighbors:
            if n not in self.safe:
                self.unsafe.add(n)
                self.frontier.add(n)

    #  Bewegungswahl basierend auf Knowledge 
    def pick_action_with_knowledge(self):
        x, y = self.pos()
        moves = {
            Action.MOVE_UP:    (x, y - 1),
            Action.MOVE_DOWN:  (x, y + 1),
            Action.MOVE_LEFT:  (x - 1, y),
            Action.MOVE_RIGHT: (x + 1, y)
        }

        valid = {
            a: p for a, p in moves.items()
            if 0 <= p[0] < self.grid_size and 0 <= p[1] < self.grid_size
        }

        # 1. Bevorzuge sichere, noch nicht besuchte Felder
        safe_moves = [a for a, p in valid.items() if p in self.safe and p not in self.visited]
        if safe_moves:
            return random.choice(safe_moves)

        # 2. Dann unbekannte Felder (weder sicher noch unsicher)
        unknown_moves = [
            a for a, p in valid.items()
            if p not in self.safe and p not in self.unsafe
        ]
        if unknown_moves:
            return random.choice(unknown_moves)

        # 3. Wenn nur noch unsichere übrig sind: irgendeinen valid Move
        return random.choice(list(valid.keys()))
