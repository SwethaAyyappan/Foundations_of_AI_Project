from opponent_model import OpponentModel

# Create an opponent model
model = OpponentModel()

# Simulate observing the opponent's actions
model.observe_action("raise")
model.observe_action("raise")
model.observe_action("call")
model.observe_action("fold")
model.observe_action("raise")
model.observe_action("raise")
#model.observe_action("bluff")
# Display observed action counts
print("Action counts:", model.action_counts)

# Calculate and display Bayesian probabilities
probabilities = model.get_action_probabilities()

print("\nEstimated action probabilities:")
for action, probability in probabilities.items():
    print(f"{action}: {probability:.2f}")

print("\nResetting opponent model...")
model.reset()

print("Action counts after reset:", model.action_counts)
print("Probabilities after reset:", model.get_action_probabilities())    