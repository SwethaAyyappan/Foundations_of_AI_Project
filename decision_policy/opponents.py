"""
opponents.py - Scripted opponents for testing the agent in Leduc Hold'em (RLCard).

Three fixed strategies. They never learn or change, so they are a known
benchmark for the adaptive agent:
  AggressiveAgent   : raises most hands (lots of bluffs), rarely folds
  ConservativeAgent : raises only strong hands, folds weak hands to a bet
  RandomAgent       : picks uniformly among the legal actions
All of them use raw action strings (use_raw = True) and only pick legal actions.
"""
import numpy as np

RANKS = {"J": 0, "Q": 1, "K": 2}


def hand_strength(raw_obs):
    """0 = weak, 1 = medium, 2 = strong, judged from the player's own cards."""
    hand, public = raw_obs["hand"], raw_obs.get("public_card")
    r = RANKS[hand[1]]
    if public is None:              # before the public card: J weak, Q medium, K strong
        return r
    if RANKS[public[1]] == r:       # pair with the public card
        return 2
    return 1 if r == 2 else 0       # K high is medium, anything lower is weak


def choose(legal, *preferences):
    """First preferred action that is legal (falls back to any legal action)."""
    for action in preferences:
        if action in legal:
            return action
    return legal[0]


class ScriptedAgent:
    """Base class: plugs into RLCard like any other agent."""
    use_raw = True
    name = "scripted"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, raw_obs, legal):
        raise NotImplementedError

    def step(self, state):
        raw_obs = state["raw_obs"]
        legal = [str(a) for a in raw_obs["legal_actions"]]
        return self.act(raw_obs, legal)

    def eval_step(self, state):
        return self.step(state), {}


class AggressiveAgent(ScriptedAgent):
    """Raises strong hands always and everything else `bluff_rate` of the time."""
    name = "aggressive"

    def __init__(self, seed=None, bluff_rate=0.7):
        super().__init__(seed)
        self.bluff_rate = bluff_rate

    def act(self, raw_obs, legal):
        strength = hand_strength(raw_obs)
        if strength == 2 or self.rng.random() < self.bluff_rate:
            return choose(legal, "raise", "call", "check")
        if "call" in legal and strength == 0 and self.rng.random() < 0.1:
            return "fold"                       # folds only rarely
        return choose(legal, "call", "check")


class ConservativeAgent(ScriptedAgent):
    """Raises only strong hands, calls medium hands, folds weak hands to a bet."""
    name = "conservative"

    def act(self, raw_obs, legal):
        strength = hand_strength(raw_obs)
        if strength == 2:
            return choose(legal, "raise", "call", "check")
        if "call" in legal:                     # facing a bet
            return "call" if strength == 1 else "fold"
        return choose(legal, "check", "call")


class RandomAgent(ScriptedAgent):
    """Uniformly random legal action (written here to avoid RLCard's torch import)."""
    name = "random"

    def act(self, raw_obs, legal):
        return str(self.rng.choice(legal))
