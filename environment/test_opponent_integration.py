import rlcard
from rlcard.agents import RandomAgent
from opponent_model.opponent_model import OpponentModel

# Create the Leduc Hold'em environment
env = rlcard.make("leduc-holdem")

# Create two random agents
agents = [
    RandomAgent(num_actions=env.num_actions),
    RandomAgent(num_actions=env.num_actions)
]

env.set_agents(agents)

# Run one complete game
trajectories, payoffs = env.run(is_training=False)

# Create our Bayesian opponent model
model = OpponentModel()

# Our AI is Player 0, so Player 1 is the opponent
opponent_trajectory = trajectories[1]

print("Opponent trajectory length:", len(opponent_trajectory))
print("Opponent trajectory contents:")

for i, item in enumerate(opponent_trajectory):
    print(i, type(item), item if isinstance(item, (int, float)) else "Game state")

# Extract the opponent's actions and update the model
for i in range(1, len(opponent_trajectory), 2):
    action_id = int(opponent_trajectory[i])
    action_name = env.actions[action_id]

    model.observe_action(action_name)

# Display results
print("Opponent action counts:", model.action_counts)

print("\nUpdated opponent probabilities:")
for action, probability in model.get_action_probabilities().items():
    print(f"{action}: {probability:.2f}")