
import rlcard
import numpy as np
from collections import defaultdict

from opponent_model.opponent_model import OpponentModel
from environment.test_opponents import ScriptedOpponentAgent


# --------------------------------------------------
# 1. Experiment configuration
# --------------------------------------------------

NUM_GAMES = 200

STYLES = [
    "aggressive",
    "conservative",
    "random"
]


# --------------------------------------------------
# 2. Run Bayesian learning experiment
# --------------------------------------------------

def run_experiment(style):

    env = rlcard.make("leduc-holdem")

    model = OpponentModel()

    random_agent = rlcard.agents.RandomAgent(
        num_actions=env.num_actions
    )

    opponent = ScriptedOpponentAgent(
        style=style,
        seed=42
    )

    env.set_agents([random_agent, opponent])

    # Actual action counts
    actual_counts = defaultdict(lambda: defaultdict(int))

    # Collect observations over multiple games
    for game in range(NUM_GAMES):

        trajectories, payoffs = env.run(
            is_training=False
        )

        trajectory = trajectories[1]

        for i in range(0, len(trajectory) - 1, 2):

            state = trajectory[i]
            action_id = trajectory[i + 1]

            raw_obs = state["raw_obs"]

            # Handle both string and integer actions
            if isinstance(action_id, (int, np.integer)):
                action = env.actions[int(action_id)]
            else:
                action = str(action_id)

            # Identify betting round
            public_card = raw_obs.get("public_card")

            round_number = (
                0 if public_card is None else 1
            )

            # Identify whether facing a bet
            facing_bet = (
                "call" in raw_obs["legal_actions"]
            )

            context = (round_number, facing_bet)

            # Record actual behavior
            actual_counts[context][action] += 1

            # Update Bayesian model
            model.observe_action(
                action,
                round_number,
                facing_bet
            )

    return model, actual_counts


# --------------------------------------------------
# 3. Compare actual and estimated probabilities
# --------------------------------------------------

def evaluate_model(model, actual_counts):

    estimates = model.get_estimates()["by_context"]

    errors = []

    print("\n--- Bayesian Estimates vs Actual Frequencies ---")

    for context, counts in sorted(actual_counts.items()):

        total = sum(counts.values())

        print(f"\nContext: {context}")
        print("Total observations:", total)

        for action, estimated_probability in estimates[context].items():

            actual_probability = (
                counts[action] / total
                if total > 0 else 0
            )

            # Absolute estimation error
            error = abs(
                estimated_probability - actual_probability
            )

            errors.append(error)

            print(
                f"{action:<6} | "
                f"Actual: {actual_probability:.3f} | "
                f"Estimated: {estimated_probability:.3f} | "
                f"Error: {error:.3f}"
            )

    mae = np.mean(errors) if errors else 0

    return mae


# --------------------------------------------------
# 4. Run all three experiments
# --------------------------------------------------

if __name__ == "__main__":

    print("STARTING BAYESIAN OPPONENT IDENTIFICATION")
    print("Games per opponent:", NUM_GAMES)

    results = {}

    for style in STYLES:

        print("\n" + "=" * 55)
        print("OPPONENT:", style.upper())
        print("=" * 55)

        model, actual_counts = run_experiment(style)

        mae = evaluate_model(
            model,
            actual_counts
        )

        results[style] = mae

        print("\nMean Absolute Error:", round(mae, 4))

    # --------------------------------------------------
    # 5. Final comparison
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("FINAL BAYESIAN MODEL EVALUATION")
    print("=" * 55)

    for style, mae in results.items():

        print(
            f"{style.capitalize():<15} "
            f"Mean Absolute Error: {mae:.4f}"
        )

    print("\nBayesian evaluation completed!")
