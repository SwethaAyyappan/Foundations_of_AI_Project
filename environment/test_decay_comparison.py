
import rlcard
import numpy as np

from opponent_model.opponent_model import OpponentModel
from decision_policy.decision_policy import AdaptivePolicyAgent
from environment.test_opponents import ScriptedOpponentAgent


NUM_GAMES = 500
SWITCH_GAME = 250
SEEDS = [11, 22, 33, 44, 55]

CONTEXT = (0, True)

TARGET = {
    "fold": 0.40,
    "call": 0.55,
    "raise": 0.05
}

THRESHOLD = 0.15
REQUIRED_CONSECUTIVE = 5


def calculate_distance(estimated):

    return sum(
        abs(estimated[action] - TARGET[action])
        for action in TARGET
    ) / 2


def update_model(model, trajectory, actions,
                 switched, tracking):

    for i in range(0, len(trajectory) - 1, 2):

        state = trajectory[i]
        action_id = trajectory[i + 1]
        raw_obs = state["raw_obs"]

        if isinstance(action_id, (int, np.integer)):
            action = actions[int(action_id)]
        else:
            action = str(action_id)

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

        if switched and (round_number, facing_bet) == CONTEXT:

            tracking["observations"] += 1

            estimates = model.get_context_probabilities(CONTEXT)

            distance = calculate_distance(estimates)

            if distance <= THRESHOLD:
                tracking["consecutive"] += 1
            else:
                tracking["consecutive"] = 0

            if (
                tracking["adaptation_lag"] is None
                and tracking["consecutive"] >= REQUIRED_CONSECUTIVE
            ):
                tracking["adaptation_lag"] = tracking["observations"]


def run_experiment(decay, seed):

    env = rlcard.make("leduc-holdem")
    env.seed(seed)

    model = OpponentModel(decay=decay)

    our_agent = AdaptivePolicyAgent(
        opponent_model=model,
        temperature=0.0
    )

    opponent = ScriptedOpponentAgent(
        style="aggressive",
        seed=seed
    )

    env.set_agents([our_agent, opponent])

    tracking = {
        "observations": 0,
        "consecutive": 0,
        "adaptation_lag": None
    }

    total_payoff = 0
    wins = 0

    for game in range(NUM_GAMES):

        if game == SWITCH_GAME:
            opponent.style = "conservative"

        trajectories, payoffs = env.run(is_training=False)

        payoff = float(payoffs[0])
        total_payoff += payoff

        if payoff > 0:
            wins += 1

        update_model(
            model,
            trajectories[1],
            env.actions,
            switched=(game >= SWITCH_GAME),
            tracking=tracking
        )

    return {
        "average_payoff": total_payoff / NUM_GAMES,
        "win_rate": wins / NUM_GAMES,
        "adaptation_lag": tracking["adaptation_lag"]
    }


def main():

    decays = [1.0, 0.95]

    results = {
        decay: []
        for decay in decays
    }

    print("=" * 65)
    print("RECENCY WEIGHTING: CONTROLLED COMPARISON")
    print("=" * 65)

    for seed in SEEDS:

        print(f"\nRunning seed {seed}...")

        for decay in decays:

            result = run_experiment(decay, seed)

            results[decay].append(result)

            print(
                f"Decay={decay}: "
                f"Payoff={result['average_payoff']:.4f}, "
                f"Win rate={result['win_rate'] * 100:.2f}%, "
                f"Adaptation lag={result['adaptation_lag']}"
            )

    print("\n" + "=" * 65)
    print("AVERAGE RESULTS ACROSS ALL SEEDS")
    print("=" * 65)

    for decay in decays:

        runs = results[decay]

        payoffs = [
            r["average_payoff"] for r in runs
        ]

        win_rates = [
            r["win_rate"] for r in runs
        ]

        lags = [
            r["adaptation_lag"]
            for r in runs
            if r["adaptation_lag"] is not None
        ]

        print(f"\nDecay factor: {decay}")

        print(
            "Average payoff:",
            round(np.mean(payoffs), 4)
        )

        print(
            "Average win rate:",
            round(np.mean(win_rates) * 100, 2),
            "%"
        )

        if lags:
            print(
                "Average adaptation lag:",
                round(np.mean(lags), 2),
                "observations"
            )
        else:
            print("Adaptation lag: Not reached in any run")

    print("\nComparison completed!")


if __name__ == "__main__":
    main()
