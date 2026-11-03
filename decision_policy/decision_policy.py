"""
decision_policy.py - Opponent-aware decision policy for Leduc Hold'em (RLCard).

Inputs : the game situation (RLCard raw_obs) + opponent estimates
         from the Bayesian opponent model.
Output : one legal action ('fold' / 'call' / 'check' / 'raise').

Idea
----
1. Hand strength: exact equity by enumerating every card the opponent could
   hold (and the public card, if it isn't out yet). The deck is J,Q,K x 2.
2. Opponent range: we assume the opponent raises with their strongest hands.
   If they raise X% of the time, a raise means "top X% of hands".
   An aggressive opponent's raise is therefore weaker than a cautious one's.
3. Fold equity: P(opponent folds to our raise) comes from the estimates.
   If they continue, we assume they fold their weakest hands and continue
   with the rest.
4. Expected value: compute EV for every legal action and pick the best.

Expected opponent-estimate format (agree on this with your partner):
    {"by_context": {(round, facing_bet): {"fold": p, "call": p, "raise": p}}}
    round: 0 = before public card, 1 = after. facing_bet: True/False.
"""
import numpy as np

RANKS = {"J": 0, "Q": 1, "K": 2}
DECK = ["SJ", "HJ", "SQ", "HQ", "SK", "HK"]          # RLCard card format
RAISE_AMOUNT = {0: 2, 1: 4}
UNIFORM = {"fold": 1 / 3, "call": 1 / 3, "raise": 1 / 3}


# ---------------- hand evaluation ----------------
def rank(card):
    return RANKS[card[1]]


def showdown(ours, theirs, public):
    """1 = we win, 0.5 = tie, 0 = we lose."""
    our_pair, their_pair = rank(ours) == rank(public), rank(theirs) == rank(public)
    if our_pair != their_pair:
        return 1.0 if our_pair else 0.0
    if rank(ours) == rank(theirs):
        return 0.5
    return 1.0 if rank(ours) > rank(theirs) else 0.0


def opp_strength(card, public):
    """How strong a holding looks to the opponent (used to order their range)."""
    if public is not None and rank(card) == rank(public):
        return 10 + rank(card)
    return rank(card)


def initial_range(hand, public):
    """All cards the opponent could hold, equally likely."""
    seen = {hand} | ({public} if public else set())
    return {c: 1.0 for c in DECK if c not in seen}


def top_fraction(rng, frac, public):
    """Keep only the strongest `frac` of the (weighted) range."""
    frac = float(np.clip(frac, 0.05, 1.0))
    total = sum(rng.values())
    budget, out = frac * total, {}
    for card in sorted(rng, key=lambda c: -opp_strength(c, public)):
        take = min(rng[card], budget)
        if take > 0:
            out[card] = take
        budget -= take
    return out


def equity(hand, public, rng):
    """P(we win) against a weighted opponent range."""
    num = den = 0.0
    for opp, w in rng.items():
        if public is not None:
            eq = showdown(hand, opp, public)
        else:  # average over every possible public card
            boards = [p for p in DECK if p not in (hand, opp)]
            eq = np.mean([showdown(hand, opp, p) for p in boards])
        num += w * eq
        den += w
    return num / den if den else 0.5


# ---------------- the policy ----------------
def get_probs(estimates, context):
    if not estimates:
        return UNIFORM
    return estimates.get("by_context", {}).get(context, UNIFORM)


def decide(raw_obs, opp_estimates=None, opp_raised=None, temperature=0.0, rng_seed=None):
    """
    raw_obs       : RLCard state['raw_obs'] for OUR player.
    opp_estimates : output of the opponent model (None = no information yet).
    opp_raised    : did the opponent raise this round? Inferred from chips if None.
    temperature   : 0 = always pick the best-EV action; >0 = softmax sampling
                    (makes the agent less predictable).
    Returns (action, info) where info explains the decision.
    """
    hand, public = raw_obs["hand"], raw_obs.get("public_card")
    legal = [str(a) for a in raw_obs["legal_actions"]]
    rnd = 0 if public is None else 1
    bet = RAISE_AMOUNT[rnd]

    my_chips = raw_obs["my_chips"]
    opp_chips = max(raw_obs["all_chips"])
    pot = sum(raw_obs["all_chips"])
    cost = max(0, opp_chips - my_chips)        # chips needed to call
    if opp_raised is None:                     # a gap of one full raise = they raised
        opp_raised = cost >= bet

    # 1) opponent's range given what they've done
    opp_range = initial_range(hand, public)
    kept = 1.0                                  # fraction of their hands still in range
    if opp_raised:
        raise_freq = np.mean([get_probs(opp_estimates, (rnd, f))["raise"] for f in (True, False)])
        kept = float(np.clip(raise_freq, 0.15, 1.0))
        opp_range = top_fraction(opp_range, kept, public)
    eq_now = equity(hand, public, opp_range)

    # 2) how they respond if WE raise (they will be facing a bet).
    #    They fold their weakest `fold_rate` of ALL hands. If their range has already
    #    been narrowed to the strongest `kept` hands, only the overlap can fold:
    #    a cautious player who raised is strong and will rarely fold now.
    fold_rate = get_probs(opp_estimates, (rnd, True))["fold"]
    p_fold = max(0.0, kept + fold_rate - 1.0) / kept
    cont_range = top_fraction(opp_range, 1 - p_fold, public)
    eq_cont = equity(hand, public, cont_range)

    # 3) EV of each legal action (chips we get back minus new chips we put in)
    ev = {}
    if "fold" in legal and "check" not in legal:   # never fold when checking is free
        ev["fold"] = 0.0
    if "check" in legal:
        ev["check"] = eq_now * pot
    if "call" in legal:
        ev["call"] = eq_now * (pot + cost) - cost
    if "raise" in legal:
        invest = cost + bet
        # opponent re-raises are folded into "continues" to keep the model simple
        ev["raise"] = p_fold * pot + (1 - p_fold) * (eq_cont * (pot + invest + bet) - invest)

    # 4) choose
    actions = list(ev)
    values = np.array([ev[a] for a in actions])
    if temperature > 0:
        p = np.exp((values - values.max()) / temperature)
        p /= p.sum()
        action = np.random.default_rng(rng_seed).choice(actions, p=p)
    else:
        action = actions[int(values.argmax())]

    info = {"equity": round(float(eq_now), 3), "equity_if_called": round(float(eq_cont), 3),
            "opp_fold_to_raise": round(float(p_fold), 3), "opp_raised": bool(opp_raised),
            "ev": {a: round(float(v), 2) for a, v in ev.items()}}
    return str(action), info


# ---------------- RLCard agent wrapper ----------------
class AdaptivePolicyAgent:
    """Plugs into RLCard. use_raw=True means step() returns action strings."""
    use_raw = True

    def __init__(self, opponent_model=None, temperature=0.0):
        self.opponent_model = opponent_model
        self.temperature = temperature

    def _estimates(self):
        return self.opponent_model.get_estimates() if self.opponent_model else None

    def step(self, state):
        return decide(state["raw_obs"], self._estimates(), temperature=self.temperature)[0]

    def eval_step(self, state):
        return decide(state["raw_obs"], self._estimates())


# ---------------- test: same situations, different opponents ----------------
def profile(fold_vs_bet, call_vs_bet, raise_vs_bet, call_no_bet, raise_no_bet):
    ctx = {}
    for r in (0, 1):
        ctx[(r, True)] = {"fold": fold_vs_bet, "call": call_vs_bet, "raise": raise_vs_bet}
        ctx[(r, False)] = {"fold": 0.0, "call": call_no_bet, "raise": raise_no_bet}
    return {"by_context": ctx}


if __name__ == "__main__":
    opponents = {
        "Unknown":      None,
        "Aggressive":   profile(0.05, 0.25, 0.70, 0.30, 0.70),
        "Conservative": profile(0.55, 0.40, 0.05, 0.90, 0.10),
    }
    situations = {
        "J, preflop, no bet to face":  dict(hand="SJ", public_card=None, all_chips=[2, 2], my_chips=2,
                                            legal_actions=["raise", "fold", "check"]),
        "Q, board K, facing a raise":  dict(hand="SQ", public_card="HK", all_chips=[4, 8], my_chips=4,
                                            legal_actions=["call", "raise", "fold"]),
        "K, board J, no bet to face":  dict(hand="SK", public_card="HJ", all_chips=[4, 4], my_chips=4,
                                            legal_actions=["raise", "fold", "check"]),
        "K, board Q, facing a raise":  dict(hand="SK", public_card="HQ", all_chips=[6, 10], my_chips=6,
                                            legal_actions=["call", "fold"]),
    }
    for name, obs in situations.items():
        print(f"\n{name}")
        for opp, est in opponents.items():
            action, info = decide(obs, est)
            print(f"  vs {opp:<12} -> {action:<5}  equity={info['equity']:.2f}  EVs={info['ev']}")
