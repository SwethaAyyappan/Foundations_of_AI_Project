
import rlcard
import numpy as np

from opponent_model.opponent_model import OpponentModel
from decision_policy.decision_policy import AdaptivePolicyAgent
from environment.test_opponents import ScriptedOpponentAgent


# --------------------------------------------------
# Experiment configuration
# --------------------------------------------------

NUM_GAMES = 500
SWITCH_GAME = 250

CONTEXT_TO_TRACK = (0, True)

# Conservative opponent's expected probabilities
# in context (0, True).
CONSERVATIVE_TARGET = {
    "fold": 0.40,
    "call": 0.55,
    "raise": 0.05
}

DISTANCE_THRESHOLD = 0.15
REQUIRED_CONSECUTIVE_OBSERVATIONS = 5


# --------------------------------------------------
# Calculate total variation distance
# --------------------------------------------------

def calculate_distance(estimated, target):

    distance = 0.0

    for action in target:
        distance += abs(
            estimated[action] - target[action]
        )

    return distance / 2


# --------------------------------------------------
# Update model and measure adaptation
# --------------------------------------------------

def update_opponent_model(model, trajectory, actions,
                          switched, tracking):

    for i in range(0, len(trajectory) - 1, 2):

        state = trajectory[i]
        action_id = trajectory[i + 1]

        raw_obs = state["raw_obs"]

        if isinstance(action_id, (int, np.integer)):
            action = actions[int(action_id)]
        else:
            action = str(action_id)

        public_card = raw_obs.get("public_card")

        round_number = (
            0 if public_card is None else 1
        )

        facing_bet = (
            "call" in raw_obs["legal_actions"]
        )

        context = (round_number, facing_bet)

        # Update the Bayesian model
        model.observe_action(
            action,
            round_number,
            facing_bet
        )

        # Measure adaptation only after the switch
        # and only in our selected context.
        if switched and context == CONTEXT_TO_TRACK:

            tracking["observations"] += 1

            estimates = model.get_context_probabilities(
                CONTEXT_TO_TRACK
            )

            distance = calculate_distance(
                estimates,
                CONSERVATIVE_TARGET
            )

            if distance <= DISTANCE_THRESHOLD:

                tracking["consecutive"] += 1

            else:
                tracking["consecutive"] = 0

            # Record the first time the model stays
            # close to the conservative target.
            if (
                tracking["adaptation_lag"] is None
                and tracking["consecutive"]
                >= REQUIRED_CONSECUTIVE_OBSERVATIONS
            ):

                tracking["adaptation_lag"] = (
                    tracking["observations"]
                )


# --------------------------------------------------
# Run strategy-switching experiment
# --------------------------------------------------

def run_strategy_switch_experiment():

    env = rlcard.make("leduc-holdem")

    # One model throughout the entire experiment
    model = OpponentModel(decay=0.95)

    our_agent = AdaptivePolicyAgent(
        opponent_model=model,
        temperature=0.0
    )

    opponent = ScriptedOpponentAgent(
        style="aggressive",
        seed=42
    )

    env.set_agents([
        our_agent,
        opponent
    ])

    tracking = {
        "observations": 0,
        "consecutive": 0,
        "adaptation_lag": None
    }

    total_payoff = 0
    wins = 0
    losses = 0
    ties = 0

    print("=" * 65)
    print("ADAPTIVE AI: OPPONENT STRATEGY-SWITCHING EXPERIMENT")
    print("=" * 65)

    print(f"Total games: {NUM_GAMES}")
    print(f"Switch occurs after game: {SWITCH_GAME}")
    print("Initial opponent style: Aggressive")
    print("New opponent style: Conservative")
    print("Decay factor:", model.decay)

    for game in range(NUM_GAMES):

        # Switch opponent's strategy halfway through
        if game == SWITCH_GAME:

            opponent.style = "conservative"

            print("\n" + "-" * 65)
            print("OPPONENT SWITCHED TO CONSERVATIVE!")
            print("-" * 65)

            print("\nEstimates immediately before switch:")

            print(
                model.get_context_probabilities(
                    CONTEXT_TO_TRACK
                )
            )

        # Play one game
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

        # Update model using opponent's actions
        update_opponent_model(
            model,
            trajectories[1],
            env.actions,
            switched=(game >= SWITCH_GAME),
            tracking=tracking
        )

        # Display estimates every 25 games after switch
        if (
            game >= SWITCH_GAME
            and (game + 1 - SWITCH_GAME) % 25 == 0
        ):

            estimates = model.get_context_probabilities(
                CONTEXT_TO_TRACK
            )

            distance = calculate_distance(
                estimates,
                CONSERVATIVE_TARGET
            )

            print(
                f"\nGame {game + 1}: "
                f"Fold={estimates['fold']:.3f}, "
                f"Call={estimates['call']:.3f}, "
                f"Raise={estimates['raise']:.3f}, "
                f"Distance={distance:.3f}"
            )

    # --------------------------------------------------
    # Final results
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    print("Average payoff:", round(total_payoff / NUM_GAMES, 4))
    print("Win rate:", round(wins / NUM_GAMES * 100, 2), "%")
    print("Wins:", wins)
    print("Losses:", losses)
    print("Ties:", ties)

    print("\nFinal opponent estimates:")

    final_estimates = model.get_estimates()["by_context"]

    for context, probabilities in final_estimates.items():

        print("\nContext:", context)

        for action, probability in probabilities.items():

            print(f"  {action}: {probability:.3f}")

    print("\n" + "-" * 65)
    print("ADAPTATION ANALYSIS")
    print("-" * 65)

    print(
        "Relevant observations after switch:",
        tracking["observations"]
    )

    if tracking["adaptation_lag"] is not None:

        print(
            "Adaptation lag:",
            tracking["adaptation_lag"],
            "relevant observations"
        )

        print(
            "The model reached the target distance "
            "for 5 consecutive observations."
        )

    else:

        print(
            "The model did not reach the adaptation "
            "threshold during this experiment."
        )

    print("\nExperiment completed!")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    run_strategy_switch_experiment()
