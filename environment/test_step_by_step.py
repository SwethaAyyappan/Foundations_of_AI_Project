import rlcard

# Initialize Leduc Hold'em
env = rlcard.make("leduc-holdem")

# Start a new game
state, player_id = env.reset()

print("Game started!")
print("Current player:", player_id)
print("Legal actions:", state["legal_actions"])

# Select the first legal action
action_id = list(state["legal_actions"].keys())[0]

print("\nTaking action:", env.actions[action_id])

# Execute one action
next_state, next_player_id = env.step(action_id)

print("\nAfter the action:")
print("Next player:", next_player_id)
print("Next player's legal actions:", next_state["legal_actions"])