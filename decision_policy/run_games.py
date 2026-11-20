"""
run_games.py - RLCard integration and game testing for the decision policy.

Contents
  FixedStrategyAgent : the decision policy with FIXED opponent estimates that
                       are never updated (the baseline to compare against)
  PlaceholderModel   : a simple counting model with the same interface as the
                       Bayesian opponent model, used only until the real model
                       is plugged in (needs update(action_record, opp_id) and
                       get_estimates())
  LegalityCheck      : wraps an agent and stops if it ever picks an illegal action
  play()             : the RLCard game loop for one matchup

Run:  python run_games.py          (2000 hands per matchup)
      python run_games.py 500      (quicker)
"""
import sys
import rlcard
from decision_policy import AdaptivePolicyAgent, label_actions
from opponents import AggressiveAgent, ConservativeAgent, RandomAgent


# ---------------- fixed-strategy agent ----------------
class FixedStrategyAgent(AdaptivePolicyAgent):
    """
    Same decision policy, but the opponent estimates never change during play.
    fixed_estimates=None means "assume every action is equally likely".
    """
    def __init__(self, fixed_estimates=None, temperature=0.0, range_trust=1.0):
        super().__init__(opponent_model=None, temperature=temperature, range_trust=range_trust)
        self.fixed_estimates = fixed_estimates

    def _estimates(self):
        return self.fixed_estimates


# ---------------- placeholder opponent model ----------------
class PlaceholderModel:
    """
    Stand-in for the Bayesian opponent model, with the same interface.
    Counts the opponent's actions in each (round, facing_bet) context, starting
    from one pseudo-count per action. Swap in the real model when it is ready.
    """
    def __init__(self):
        self.counts = {}
        for r in (0, 1):
            self.counts[(r, True)] = {"fold": 1, "call": 1, "raise": 1}
            self.counts[(r, False)] = {"fold": 1, "check": 1, "raise": 1}

    def update(self, action_record, opp_id):
        for pid, action, r, facing in label_actions(action_record):
            if pid == opp_id and action in self.counts[(r, facing)]:
                self.counts[(r, facing)][action] += 1

    def get_estimates(self):
        by_context = {}
        for ctx, c in self.counts.items():
            total = sum(c.values())
            by_context[ctx] = {a: n / total for a, n in c.items()}
        return {"by_context": by_context}


# ---------------- game loop ----------------
class LegalityCheck:
    """Wraps an agent and raises an error if it returns an illegal action."""
    def __init__(self, agent):
        self.agent = agent
        self.use_raw = agent.use_raw

    def _check(self, state, action):
        legal = [str(a) for a in state.get("raw_legal_actions", state["raw_obs"]["legal_actions"])]
        if str(action) not in legal:
            raise RuntimeError(f"illegal action {action!r}, legal actions were {legal}")

    def step(self, state):
        action = self.agent.step(state)
        self._check(state, action)
        return action

    def eval_step(self, state):
        action, info = self.agent.eval_step(state)
        self._check(state, action)
        return action, info


def play(agent, opponent, n_hands, seed=0, model=None):
    """
    Plays n_hands of Leduc Hold'em between agent and opponent, swapping seats
    every hand. If a model is given, it is updated with the opponent's actions
    after each hand. Returns (average payoff per hand for agent, errors).
    """
    env = rlcard.make("leduc-holdem", config={"seed": seed})
    total, completed, errors = 0.0, 0, 0
    for i in range(n_hands):
        seat = i % 2
        seats = [None, None]
        seats[seat], seats[1 - seat] = LegalityCheck(agent), LegalityCheck(opponent)
        env.set_agents(seats)
        try:
            trajectories, payoffs = env.run(is_training=False)
        except Exception as e:                       # a crash or an illegal action
            errors += 1
            if errors <= 3:
                print(f"    hand {i}: {type(e).__name__}: {e}")
            continue
        if abs(sum(payoffs)) > 1e-9:                 # chips must be zero-sum
            errors += 1
            print(f"    hand {i}: payoffs do not sum to zero: {payoffs}")
            continue
        total += payoffs[seat]
        completed += 1
        if model is not None:
            record = getattr(env, "action_recorder", None) or trajectories[seat][-1]["action_record"]
            model.update(record, 1 - seat)
    return total / max(1, completed), errors


# ---------------- game testing ----------------
OPPONENTS = {"aggressive": AggressiveAgent, "conservative": ConservativeAgent, "random": RandomAgent}


def main(n_hands=2000):
    print(f"{n_hands} hands per matchup, seats swapped every hand.")
    print("Numbers are the agent's average payoff per hand (RLCard units, positive = winning).\n")
    print(f"{'opponent':<14}{'fixed':>10}{'adaptive':>10}{'errors':>8}")
    total_errors = 0
    for name, Opponent in OPPONENTS.items():
        fixed, e1 = play(FixedStrategyAgent(), Opponent(seed=1), n_hands, seed=1)
        model = PlaceholderModel()
        adaptive, e2 = play(AdaptivePolicyAgent(model), Opponent(seed=1), n_hands, seed=1, model=model)
        total_errors += e1 + e2
        print(f"{name:<14}{fixed:>10.3f}{adaptive:>10.3f}{e1 + e2:>8}")
    print()
    if total_errors == 0:
        print(f"All {6 * n_hands} games completed without errors.")
    else:
        print(f"{total_errors} games had errors - see the messages above.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
