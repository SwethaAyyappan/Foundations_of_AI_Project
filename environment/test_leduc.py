
import rlcard
from rlcard.agents import RandomAgent
from opponent_model.opponent_model import OpponentModel


# --------------------------------------------------
# 1. Initialize environment and opponent model
# --------------------------------------------------

env = rlcard.make("leduc-holdem")
model = OpponentModel()

print("Leduc Hold'em environment initialized!")


# --------------------------------------------------
# 2. Create two random agents
# --------------------------------------------------

agent1 = RandomAgent(num_actions=env.num_actions)
agent2 = RandomAgent(num_actions=env.num_actions)

env.set_agents([agent1, agent2])


# --------------------------------------------------
# 3. Run one complete game
# --------------------------------------------------

trajectories, payoffs = env.run(is_training=False)


# --------------------------------------------------
# 4. Display game results
# --------------------------------------------------

print("\n--- Game Finished ---")
print("Player 0 payoff:", payoffs[0])
print("Player 1 payoff:", payoffs[1])


# --------------------------------------------------
# 5. Observe Player 1's actions and inspect observations
# --------------------------------------------------

print("\n--- Tracking Opponent Actions ---")

trajectory = trajectories[1]

for i in range(0, len(trajectory) - 1, 2):

    state = trajectory[i]
    action_id = trajectory[i + 1]

    raw_obs = state["raw_obs"]

    
    # Convert action ID to action name
    try:
        action = env.actions[int(action_id)]
    except (ValueError, TypeError):
        action = action_id    

    # Identify betting round
    public_card = raw_obs.get("public_card")
    round_number = 0 if public_card is None else 1

    # Get legal actions
    legal_actions = raw_obs.get("legal_actions", [])
    raw_legal_actions = state.get("raw_legal_actions", [])

    # Identify whether opponent is facing a bet
    facing_bet = "call" in legal_actions

    # Update Bayesian opponent model
    model.observe_action(
        action=action,
        round_number=round_number,
        facing_bet=facing_bet
    )

    print("\n-----------------------------------")
    print("Opponent action:", action)
    print("Betting round:", round_number)
    print("Facing a bet:", facing_bet)

    # NEW: Inspect actual RLCard observation
    print("\n--- Raw Observation Details ---")

    print("Hand:", raw_obs.get("hand"))
    print("Public card:", raw_obs.get("public_card"))
    print("My chips:", raw_obs.get("my_chips"))
    print("All chips:", raw_obs.get("all_chips"))
    print("Current player:", raw_obs.get("current_player"))

    print("Legal actions:", legal_actions)
    print("Raw legal actions:", raw_legal_actions)

    # NEW: Print all available observation keys
    print("\nAvailable raw_obs keys:", list(raw_obs.keys()))


# --------------------------------------------------
# 6. Display learned action counts
# --------------------------------------------------

print("\n--- Opponent Action Counts ---")

for context, counts in model.get_action_counts().items():
    print(context, counts)


# --------------------------------------------------
# 7. Display Bayesian estimates
# --------------------------------------------------

print("\n--- Learned Opponent Behavior ---")

estimates = model.get_estimates()

for context, probabilities in estimates["by_context"].items():

    print(f"\nContext {context}:")

    for action, probability in probabilities.items():
        print(f"{action}: {probability:.3f}")


# --------------------------------------------------
# 8. Display overall opponent probabilities
# --------------------------------------------------

print("\n--- Overall Opponent Probabilities ---")

for action, probability in model.get_action_probabilities().items():
    print(f"{action}: {probability:.3f}")

print("\nGame and opponent modeling completed!")



