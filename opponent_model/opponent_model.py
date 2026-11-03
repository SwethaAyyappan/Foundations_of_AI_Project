class OpponentModel:

    def __init__(self):
        # Track the number of times the opponent takes each action
        self.action_counts = {
            "call": 0,
            "raise": 0,
            "fold": 0,
            "check": 0
        }

    def observe_action(self, action):
        if action not in self.action_counts:
            raise ValueError(f"Invalid action: {action}")

        self.action_counts[action] += 1
    def get_action_probabilities(self):
        alpha = 1

        total_actions = sum(self.action_counts.values())
        num_actions = len(self.action_counts)

        probabilities = {}

        for action, count in self.action_counts.items():
            probabilities[action] = (
                (count + alpha) /
                (total_actions + alpha * num_actions)
            )

        return probabilities

    def reset(self):
        self.action_counts = {
            "call": 0,
            "raise": 0,
            "fold": 0,
            "check": 0
        }

