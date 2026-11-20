
import rlcard
import numpy as np

from rlcard.agents import RandomAgent

from opponent_model.opponent_model import OpponentModel
from decision_policy.decision_policy import AdaptivePolicyAgent
from environment.test_opponents import ScriptedOpponentAgent


# --------------------------------------------------
# Experiment configuration
# --------------------------------------------------

NUM_GAMES = 500

OPPONENT_STYLES = [
    "aggressive",
    "conservative",
    "random"
]


# --------------------------------------------------
# Update Bayesian model using opponent actions
# --------------------------------------------------

def update_opponent_model(model, trajectory, actions):

    for i in range(0, len(trajectory) - 1, 2):

        state = trajectory[i]
        action_id = trajectory[i + 1]

        raw_obs = state["raw_obs"]

        # Support both integer IDs and raw action strings
        if isinstance(action_id, (int, np.integer)):
            action = actions[int(action_id)]
        else:
            action = str(action_id)


        try:
            action = actions[int(action_id)]
        except (ValueError, TypeError):
            action = str(action_id)    




        # Identify betting round
        public_card = raw_obs.get("public_card")

        round_number = (
            0 if public_card is None else 1
        )

        # Identify whether opponent faces a bet
        facing_bet = (
            "call" in raw_obs["legal_actions"]
        )

        # Update Bayesian model
        model.observe_action(
            action,
            round_number,
            facing_bet
        )


# --------------------------------------------------
# Run one experiment
# --------------------------------------------------

def run_experiment(style, adaptive):

    env = rlcard.make("leduc-holdem")

    # Create Bayesian model only for adaptive agent
    model = OpponentModel() if adaptive else None

    our_agent = AdaptivePolicyAgent(
        opponent_model=model,
        temperature=0.0
    )

    opponent = ScriptedOpponentAgent(
        style=style,
        seed=42
    )

    env.set_agents([
        our_agent,
        opponent
    ])

    total_payoff = 0
    wins = 0
    losses = 0
    ties = 0

    for game in range(NUM_GAMES):

        trajectories, payoffs = env.run(
            is_training=False
        )

        # Our AI is Player 0
        payoff = float(payoffs[0])

        total_payoff += payoff

        if payoff > 0:
            wins += 1

        elif payoff < 0:
            losses += 1

        else:
            ties += 1

        # Adaptive agent learns from Player 1
        if adaptive:

            update_opponent_model(
                model,
                trajectories[1],
                env.actions
            )

    results = {
        "average_payoff": total_payoff / NUM_GAMES,
        "win_rate": wins / NUM_GAMES,
        "wins": wins,
        "losses": losses,
        "ties": ties
    }

    return results, model


# --------------------------------------------------
# Display experiment results
# --------------------------------------------------

def display_results(style, fixed, adaptive):

    print("\nOpponent:", style.upper())

    print("-" * 60)

    print(
        f"{'Metric':<20}"
        f"{'Fixed':<18}"
        f"{'Adaptive':<18}"
    )

    print("-" * 60)

    print(
        f"{'Average payoff':<20}"
        f"{fixed['average_payoff']:<18.4f}"
        f"{adaptive['average_payoff']:<18.4f}"
    )

    print(
        f"{'Win rate':<20}"
        f"{fixed['win_rate'] * 100:<18.2f}"
        f"{adaptive['win_rate'] * 100:<18.2f}"
    )

    print(
        f"{'Wins':<20}"
        f"{fixed['wins']:<18}"
        f"{adaptive['wins']:<18}"
    )

    print(
        f"{'Losses':<20}"
        f"{fixed['losses']:<18}"
        f"{adaptive['losses']:<18}"
    )

    print(
        f"{'Ties':<20}"
        f"{fixed['ties']:<18}"
        f"{adaptive['ties']:<18}"
    )

    difference = (
        adaptive["average_payoff"]
        - fixed["average_payoff"]
    )

    print("-" * 60)
    print("Adaptive payoff improvement:", round(difference, 4))


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("FIXED VS ADAPTIVE AI EXPERIMENT")
    print("Games per condition:", NUM_GAMES)
    print("=" * 60)

    all_results = {}

    for style in OPPONENT_STYLES:

        print("\nRunning against:", style)

        # Fixed agent
        print("  Testing fixed agent...")

        fixed_results, _ = run_experiment(
            style=style,
            adaptive=False
        )

        # Adaptive agent
        print("  Testing adaptive agent...")

        adaptive_results, learned_model = run_experiment(
            style=style,
            adaptive=True
        )

        all_results[style] = {
            "fixed": fixed_results,
            "adaptive": adaptive_results
        }

        display_results(
            style,
            fixed_results,
            adaptive_results
        )

        # Show what the adaptive model learned
        print("\nLearned opponent probabilities:")

        for context, probabilities in (
            learned_model.get_estimates()["by_context"].items()
        ):

            print("\nContext:", context)

            for action, probability in probabilities.items():

                print(
                    f"  {action}: {probability:.3f}"
                )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL EXPERIMENT SUMMARY")
    print("=" * 60)

    print(
        f"{'Opponent':<18}"
        f"{'Fixed payoff':<18}"
        f"{'Adaptive payoff':<18}"
    )

    for style, results in all_results.items():

        print(
            f"{style:<18}"
            f"{results['fixed']['average_payoff']:<18.4f}"
            f"{results['adaptive']['average_payoff']:<18.4f}"
        )

    print("\nAll experiments completed!")
