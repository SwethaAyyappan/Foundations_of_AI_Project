
class OpponentModel:
    """
    Bayesian opponent model for Leduc Hold'em.

    Tracks opponent actions separately by:
    - Betting round (0 or 1)
    - Whether the opponent is facing a bet

    Uses Laplace smoothing and recency weighting.
    """

    def __init__(self, alpha=1.0, decay=0.95):

        if alpha <= 0:
            raise ValueError("Alpha must be greater than zero.")

        if not 0 < decay <= 1:
            raise ValueError("Decay must be greater than 0 and at most 1.")

        self.alpha = alpha
        self.decay = decay

        self.action_counts = {
            (0, True): {
                "fold": 0.0,
                "call": 0.0,
                "raise": 0.0
            },

            (0, False): {
                "fold": 0.0,
                "check": 0.0,
                "raise": 0.0
            },

            (1, True): {
                "fold": 0.0,
                "call": 0.0,
                "raise": 0.0
            },

            (1, False): {
                "fold": 0.0,
                "check": 0.0,
                "raise": 0.0
            }
        }

    def observe_action(self, action, round_number, facing_bet):

        if round_number not in (0, 1):
            raise ValueError("Round number must be 0 or 1.")

        if not isinstance(facing_bet, bool):
            raise TypeError("facing_bet must be True or False.")

        context = (round_number, facing_bet)

        if action not in self.action_counts[context]:
            raise ValueError(
                f"Action '{action}' is invalid for context {context}."
            )

        counts = self.action_counts[context]

        # Give less importance to older observations.
        for previous_action in counts:
            counts[previous_action] *= self.decay

        # Record the newly observed action.
        counts[action] += 1.0

    def get_context_probabilities(self, context):

        if context not in self.action_counts:
            raise ValueError(f"Invalid context: {context}")

        counts = self.action_counts[context]

        total_actions = sum(counts.values())
        num_actions = len(counts)

        probabilities = {}

        for action, count in counts.items():

            probabilities[action] = (
                (count + self.alpha) /
                (total_actions + self.alpha * num_actions)
            )

        return probabilities

    def get_estimates(self):

        estimates = {"by_context": {}}

        for context in self.action_counts:

            probabilities = self.get_context_probabilities(context)

            # Provide a consistent structure for the decision policy.
            for action in ["fold", "call", "raise", "check"]:
                if action not in probabilities:
                    probabilities[action] = 0.0

            estimates["by_context"][context] = probabilities

        return estimates

    def get_action_probabilities(self):

        totals = {
            "call": 0.0,
            "raise": 0.0,
            "fold": 0.0,
            "check": 0.0
        }

        for context_counts in self.action_counts.values():

            for action, count in context_counts.items():
                totals[action] += count

        total = sum(totals.values())
        num_actions = len(totals)

        probabilities = {}

        for action, count in totals.items():

            probabilities[action] = (
                (count + self.alpha) /
                (total + self.alpha * num_actions)
            )

        return probabilities

    def get_action_counts(self):

        return {
            context: counts.copy()
            for context, counts in self.action_counts.items()
        }

    def reset(self):

        for context_counts in self.action_counts.values():

            for action in context_counts:
                context_counts[action] = 0.0
