
import rlcard
from rlcard.agents import RandomAgent

from opponent_model.opponent_model import OpponentModel
from decision_policy.decision_policy import AdaptivePolicyAgent


NUM_GAMES = 500


def update_opponent_model(model, trajectory, actions):
    """Learn from the opponent's actions in a completed game."""

    for i in range(0, len(trajectory) - 1, 2):
        state = trajectory[i]
        action_id = trajectory[i + 1]

        raw_obs = state["raw_obs"]
        action = actions[int(action_id)]

        round_number = (
            0 if raw_obs.get("public_card") is None else 1
        )

        facing_bet = (
            "call" in raw_obs["legal_actions"]
        )

        model.observe_action(
            action,
            round_number,
            facing_bet
        )


def run_experiment(adaptive, num_games=NUM_GAMES):

    env = rlcard.make("leduc-holdem")

    # Adaptive agent learns; fixed agent does not.
    model = OpponentModel() if adaptive else None

    our_agent = AdaptivePolicyAgent(
        opponent_model=model
    )

    opponent = RandomAgent(
        num_actions=env.num_actions
    )

    env.set_agents([our_agent, opponent])

    total_payoff = 0
    wins = 0
    losses = 0
    ties = 0

    for game in range(num_games):

        trajectories, payoffs = env.run(
            is_training=False
        )

        payoff = float(payoffs[0])
        total_payoff += payoff

        if payoff > 0:
            wins += 1
        elif payoff < 0:
            losses += 1
        else:
            ties += 1

        # Learn from Player 1 after each completed game.
        if adaptive:
            update_opponent_model(
                model,
                trajectories[1],
                env.actions
            )

    results = {
        "average_payoff": total_payoff / num_games,
        "win_rate": wins / num_games,
        "wins": wins,
        "losses": losses,
        "ties": ties
    }

    return results, model


def display_results(name, results):

    print(f"\n--- {name} ---")
    print("Average payoff:", round(results["average_payoff"], 4))
    print("Win rate:", round(results["win_rate"] * 100, 2), "%")
    print("Wins:", results["wins"])
    print("Losses:", results["losses"])
    print("Ties:", results["ties"])


if __name__ == "__main__":

    print("Running fixed-agent experiment...")
    fixed_results, _ = run_experiment(
        adaptive=False
    )

    print("Running adaptive-agent experiment...")
    adaptive_results, model = run_experiment(
        adaptive=True
    )

    display_results("Fixed Agent", fixed_results)
    display_results("Adaptive Agent", adaptive_results)

    print("\n--- Learned Opponent Probabilities ---")

    for context, probabilities in model.get_estimates()["by_context"].items():
        print(f"\nContext {context}:")
        for action, probability in probabilities.items():
            print(f"  {action}: {probability:.3f}")

    print("\nExperiment completed!")
