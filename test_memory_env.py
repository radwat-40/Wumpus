from agents.agent1 import MarcAgent
from environment.env_memory import WumpusMemoryEnv

agent = MarcAgent(0, 0, "A1")
env = WumpusMemoryEnv(agent=agent, grid_size=8, num_pits=2, num_wumpus=1, num_gold=1)

obs = env.reset()
print("OBS SHAPE:", obs.shape)
