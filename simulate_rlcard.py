
import random
import rlcard

from opponent_model.opponent_model import OpponentModel




def get_context(env, state):

    raw_obs = state["raw_obs"]

    # Determine betting round
    round_number = (
        0 if raw_obs["public_card"] is None else 1
    )

    # Determine whether the player needs to match
    # the highest current chip contribution.
    my_chips = raw_obs["my_chips"]
    highest_chips = max(raw_obs["all_chips"])

    facing_bet = highest_chips > my_chips

    return round_number, facing_bet



def simulate_games(num_games=100):

    env = rlcard.make("leduc-holdem")

    # Create one model that learns across all games.
    model = OpponentModel()

    opponent_id = 1

    total_observations = 0
    total_payoff = 0

    for game_number in range(num_games):

        state, current_player = env.reset()

        # Continue until the current game ends.
        while not env.is_over():

            legal_action_ids = list(
                state["legal_actions"].keys()
            )

            # Randomly select one legal action.
            action_id = random.choice(legal_action_ids)

            # Observe Player 1's actions.
            if current_player == opponent_id:

                action_name = env.actions[int(action_id)]

                round_number, facing_bet = get_context(
                    env, state
                )

                print("\n--- OBSERVATION ---")
                print("Game:", game_number + 1)
                print("Player:", current_player)
                print("Action:", action_name)
                print("Round:", round_number)
                print("Facing bet:", facing_bet)

                model.observe_action(
                    action_name,
                    round_number,
                    facing_bet
                )

                total_observations += 1

            # Execute the action in RLCard.
            state, current_player = env.step(action_id)

        # Collect Player 1's payoff.
        payoffs = env.get_payoffs()
        total_payoff += payoffs[opponent_id]

    # Display results after all games.
    print("\n================================")
    print("SIMULATION RESULTS")
    print("================================")

    print(f"\nCompleted games: {num_games}")

    print("\nTotal opponent actions observed:")
    print(total_observations)

    print("\nOpponent action counts:")

    for context, counts in model.get_action_counts().items():
        print(f"\nContext {context}: {counts}")

    print("\nBayesian action probability estimates:")

    estimates = model.get_estimates()

    for context, probabilities in estimates["by_context"].items():

        print(f"\nContext {context}:")

        for action, probability in probabilities.items():
            print(f"{action}: {probability:.3f}")

    print("\nOverall action probabilities:")

    for action, probability in model.get_action_probabilities().items():
        print(f"{action}: {probability:.3f}")

    print("\nPlayer 1 average payoff per game:")
    print(total_payoff / num_games)


if __name__ == "__main__":
    simulate_games(num_games=100)
