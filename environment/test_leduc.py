
import rlcard
from rlcard.agents import RandomAgent

# 1. Create the Leduc Hold'em environment
env = rlcard.make("leduc-holdem")

print("Leduc Hold'em environment initialized!")

# 2. Initialize the game and inspect the first player's observation
state, player_id = env.reset()

print("\n--- Initial Game State ---")
print("Current player:", player_id)
print("Observation:", state["obs"])
print("Legal actions:", state["legal_actions"])

# 3. Create two random agents
agent1 = RandomAgent(num_actions=env.num_actions)
agent2 = RandomAgent(num_actions=env.num_actions)

# 4. Assign the agents to the environment
env.set_agents([agent1, agent2])

# 5. Run one complete game
trajectories, payoffs = env.run(is_training=False)

# 6. Display the results
print("\n--- Game Finished ---")
print("Player 0 payoff:", payoffs[0])
print("Player 1 payoff:", payoffs[1])

print("\nNumber of actions recorded:")
for player in range(env.num_players):
    print(f"Player {player}: {len(trajectories[player]) // 2} transitions")

print("\nLeduc Hold'em simulation completed successfully!")
