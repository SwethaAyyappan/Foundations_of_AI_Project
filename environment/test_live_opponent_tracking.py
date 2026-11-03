import rlcard
import random

from opponent_model.opponent_model import OpponentModel

# Initialize Leduc Hold'em
env = rlcard.make("leduc-holdem")

# Start a new game
state, current_player = env.reset()

# Create our Bayesian opponent model
model = OpponentModel()

# Player 1 is our opponent
opponent_id = 1

print("Game started!")

# Continue until the game ends
while not env.is_over():

    # Select a random legal action
    legal_actions = list(state["legal_actions"].keys())
    action_id = random.choice(legal_actions)

    # Identify the action
    action_name = env.actions[action_id]

    print(f"\nPlayer {current_player} chooses: {action_name}")

    # If the opponent acts, update our model immediately
    if current_player == opponent_id:

        model.observe_action(action_name)

        print("Opponent model updated!")
        print("Current estimates:")

        for action, probability in model.get_action_probabilities().items():
            print(f"{action}: {probability:.2f}")

    # Execute the action and move to the next player
    state, current_player = env.step(action_id)

# Display final results
print("\n--- Game Finished ---")
print("Final opponent action counts:", model.action_counts)
print("Final opponent probabilities:", model.get_action_probabilities())
print("Game payoffs:", env.get_payoffs())