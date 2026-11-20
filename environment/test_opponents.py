
import rlcard
import numpy as np
from collections import defaultdict
from rlcard.agents import RandomAgent


# --------------------------------------------------
# 1. Scripted opponent agent
# --------------------------------------------------

class ScriptedOpponentAgent:

    use_raw = True

    def __init__(self, style="aggressive", seed=42):
        self.style = style
        self.rng = np.random.default_rng(seed)

    def set_style(self, style):
        self.style = style

    def step(self, state):

        raw_obs = state["raw_obs"]
        legal_actions = raw_obs["legal_actions"]

        facing_bet = "call" in legal_actions

        # Aggressive opponent
        if self.style == "aggressive":

            if facing_bet:
                probabilities = {
                    "fold": 0.05,
                    "call": 0.30,
                    "raise": 0.65
                }
            else:
                probabilities = {
                    "fold": 0.05,
                    "check": 0.25,
                    "raise": 0.70
                }

        # Conservative opponent
        elif self.style == "conservative":

            if facing_bet:
                probabilities = {
                    "fold": 0.40,
                    "call": 0.55,
                    "raise": 0.05
                }
            else:
                probabilities = {
                    "fold": 0.10,
                    "check": 0.80,
                    "raise": 0.10
                }

        # Random opponent
        elif self.style == "random":

            probabilities = {
                action: 1.0 / len(legal_actions)
                for action in legal_actions
            }

        else:
            raise ValueError("Unknown opponent style")

        # Keep only legal actions
        actions = [
            action for action in probabilities
            if action in legal_actions
        ]

        weights = np.array(
            [probabilities[action] for action in actions],
            dtype=float
        )

        # Normalize probabilities
        weights /= weights.sum()

        return self.rng.choice(actions, p=weights)

    def eval_step(self, state):
        action = self.step(state)
        return action, {}


# --------------------------------------------------
# 2. Run experiments against different opponents
# --------------------------------------------------

NUM_GAMES = 200

STYLES = ["aggressive", "conservative", "random"]


def run_experiment(style):

    env = rlcard.make("leduc-holdem")

    random_agent = RandomAgent(
        num_actions=env.num_actions
    )

    opponent = ScriptedOpponentAgent(
        style=style
    )

    env.set_agents([random_agent, opponent])

    action_counts = defaultdict(lambda: defaultdict(int))

    for game in range(NUM_GAMES):

        trajectories, payoffs = env.run(
            is_training=False
        )

        # Observe Player 1's actions
        trajectory = trajectories[1]

        for i in range(0, len(trajectory) - 1, 2):

            state = trajectory[i]
            action_id = trajectory[i + 1]

            raw_obs = state["raw_obs"]

            # Handle both integer IDs and string actions
            if isinstance(action_id, (int, np.integer)):
                action = env.actions[int(action_id)]
            else:
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

            context = (round_number, facing_bet)

            # Record action frequency
            action_counts[context][action] += 1

    return action_counts


# --------------------------------------------------
# 3. Display results
# --------------------------------------------------

if __name__ == "__main__":

    for style in STYLES:

        print("\n" + "=" * 45)
        print("OPPONENT STYLE:", style.upper())
        print("=" * 45)

        results = run_experiment(style)

        for context, counts in sorted(results.items()):

            total = sum(counts.values())

            print(f"\nContext {context}")
            print("Total observations:", total)

            for action, count in sorted(counts.items()):

                probability = count / total

                print(
                    f"{action}: {count} "
                    f"({probability:.3f})"
                )

    print("\nAll opponent experiments completed!")
