from opponent_model.opponent_model import OpponentModel

# Create an opponent model
model = OpponentModel()

# Simulate observations in different contexts

# First round, facing a bet
model.observe_action("raise", 0, True)
model.observe_action("raise", 0, True)
model.observe_action("call", 0, True)

# Second round, facing a bet
model.observe_action("call", 1, True)
model.observe_action("fold", 1, True)

# First round, not facing a bet
model.observe_action("raise", 0, False)
model.observe_action("check", 0, False)

# Display action counts
print("Action counts:")
print(model.get_action_counts())

# Display context-specific Bayesian probabilities
print("\nEstimated action probabilities:")

estimates = model.get_estimates()

for context, probabilities in estimates["by_context"].items():
    print(f"\nContext {context}:")
    for action, probability in probabilities.items():
        print(f"{action}: {probability:.3f}")

# Display overall action probabilities
print("\nOverall action probabilities:")
for action, probability in model.get_action_probabilities().items():
    print(f"{action}: {probability:.3f}")

# Reset the model
print("\nResetting opponent model...")
model.reset()

print("Action counts after reset:")
print(model.get_action_counts())
