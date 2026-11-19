"""
decision_policy.py - Opponent-aware decision policy for Leduc Hold'em (RLCard).

Inputs : the game situation (RLCard state) + opponent estimates from the
         Bayesian opponent model.
Output : one legal action ('fold' / 'call' / 'check' / 'raise').

Idea
----
1. Hand strength: exact equity by enumerating every card the opponent could
   hold (and the public card, if it isn't out yet). The deck is J,Q,K x 2.
2. Opponent range: each opponent raise narrows their range to their strongest
   hands. If they raise X% of the time in that context, a raise means
   "top X% of hands". `range_trust` < 1 softens this for bluff-heavy opponents.
3. Fold equity: P(opponent folds to our raise) comes from the estimates.
   They fold the weakest hands, so the hands that continue are stronger.
4. Expected value: compute EV for every legal action and pick the best.

Agreed opponent-estimate format
-------------------------------
    {
      "by_context": {(round, facing_bet): {"fold": p, "check": p, "call": p, "raise": p}},
      "overall":    {"fold": p, "check": p, "call": p, "raise": p},   # optional fallback
    }
    round      : 0 = before the public card, 1 = after it.
    facing_bet : True if the opponent had fewer chips in than us when acting.
                 In round 0 this includes the small blind's first action.
    Missing actions count as 0 and probabilities are renormalised.
    "check" and "call" are merged into one passive action: they are never legal
    at the same time (check when not facing a bet, call when facing one).

    Both modules should label contexts with label_actions() below, so the model
    is trained on exactly the contexts the policy asks about.

RLCard rules this file relies on (rlcard/games/leducholdem)
------------------------------------------------------------
    Blinds 1 and 2; the small blind acts first in round 0.
    Raise size 2 in round 0, 4 in round 1; at most 2 raises per round.
    Round ends when two consecutive call/check actions happen (counter reset by
    a raise). Fold is always legal; check only when not behind; call only when
    behind. Pair with the public card beats any non-pair; otherwise higher rank
    wins; same rank ties.
"""
import numpy as np

RANKS = {"J": 0, "Q": 1, "K": 2}
DECK = ["SJ", "HJ", "SQ", "HQ", "SK", "HK"]          # RLCard card format
RAISE_AMOUNT = {0: 2, 1: 4}
MAX_RAISES = 2
DEFAULT = {"fold": 1 / 3, "passive": 1 / 3, "raise": 1 / 3}
MIN_RAISE_RANGE = 0.15                                # never assume a raise is only the nuts


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
    """Keep only the strongest `frac` (0..1) of the weighted range."""
    frac = float(np.clip(frac, 0.0, 1.0))
    budget, out = frac * sum(rng.values()), {}
    for card in sorted(rng, key=lambda c: -opp_strength(c, public)):
        take = min(rng[card], budget)
        if take > 1e-12:
            out[card] = take
        budget -= take
    return out


def subtract(rng, part):
    """rng minus part, dropping cards with no weight left."""
    out = {c: w - part.get(c, 0.0) for c, w in rng.items()}
    return {c: w for c, w in out.items() if w > 1e-12}


def mix(a, b, trust):
    """trust * a + (1 - trust) * b, both normalised first."""
    sa, sb = sum(a.values()), sum(b.values())
    return {c: trust * a.get(c, 0.0) / sa + (1 - trust) * b[c] / sb for c in b}


def equity(hand, public, rng):
    """Expected share of the pot against a weighted opponent range."""
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


# ---------------- reading the opponent model ----------------
def get_probs(estimates, context):
    """
    Returns {"fold", "passive", "raise"} for a (round, facing_bet) context.
    Accepts the 4-action format (check and call merged into passive) and the
    older 3-action format. Falls back to "overall", then to uniform.
    """
    p = None
    if estimates:
        p = estimates.get("by_context", {}).get(context) or estimates.get("overall")
    if not p:
        return DEFAULT
    fold = float(p.get("fold", 0.0))
    passive = float(p.get("call", 0.0)) + float(p.get("check", 0.0))
    rse = float(p.get("raise", 0.0))
    total = fold + passive + rse
    if total <= 0:
        return DEFAULT
    return {"fold": fold / total, "passive": passive / total, "raise": rse / total}


def label_actions(action_record):
    """
    Replays RLCard's action_record [(player_id, action), ...] and returns
    [(player_id, action, round, facing_bet), ...]. Mirrors leducholdem/round.py:
      raise      -> next player faces a bet, counter = 1
      call/check -> next player does not face a bet, counter += 1
      counter == 2 ends the round; fold ends the hand.
    Use this in the opponent model too, so both sides agree on contexts.
    """
    out, rnd, facing, counter = [], 0, True, 0      # small blind faces the big blind
    for pid, action in action_record:
        action = str(action)
        out.append((pid, action, rnd, facing))
        if action == "fold":
            break
        if action == "raise":
            facing, counter = True, 1
        else:
            facing, counter = False, counter + 1
        if counter >= 2:
            rnd, facing, counter = rnd + 1, False, 0
    return out


def opponent_history(raw_obs, action_record, rnd, cost, bet):
    """
    Returns (opp_raises, opp_acted_this_round), where opp_raises is a list of
    (round, facing_bet) contexts in which the opponent raised.
    Uses action_record when available; otherwise infers from chips, which only
    sees the current round and cannot tell which context the raise was in.
    """
    if action_record:
        labelled = label_actions(action_record)
        me = raw_obs.get("current_player")
        opp = (1 - me) if me is not None else labelled[-1][0]   # turns alternate
        raises = [(r, f) for pid, a, r, f in labelled if pid == opp and a == "raise"]
        acted = any(pid == opp and r == rnd for pid, a, r, f in labelled)
        return raises, acted
    # Chip fallback. Facing a gap of one full raise means they raised this round
    # (the small blind's first decision faces a gap of 1, which is not a raise).
    raises = [(rnd, None)] if cost >= bet else []
    acted = rnd == 0 or cost > 0
    return raises, acted


# ---------------- the policy ----------------
def decide(raw_obs, opp_estimates=None, action_record=None, temperature=0.0,
           range_trust=1.0, rng_seed=None):
    """
    raw_obs       : RLCard state['raw_obs'] for OUR player.
    opp_estimates : output of the opponent model (None = no information yet).
    action_record : RLCard state['action_record'] (recommended; chip inference
                    is used if it is missing).
    temperature   : 0 = always pick the best-EV action; >0 = softmax sampling.
    range_trust   : 1 = fully trust "raise means strong"; 0 = ignore raises
                    when estimating their hand.
    Returns (action, info) where info explains the decision.
    """
    hand, public = raw_obs["hand"], raw_obs.get("public_card")
    legal = [str(a) for a in raw_obs["legal_actions"]]
    rnd = 0 if public is None else 1
    bet = RAISE_AMOUNT[rnd]

    pot = sum(raw_obs["all_chips"])            # all_chips = in_chips of every player
    my_chips = raw_obs["my_chips"]
    opp_chips = pot - my_chips                 # heads-up: the rest is theirs
    cost = max(0, opp_chips - my_chips)        # chips needed to call

    opp_raises, opp_acted = opponent_history(raw_obs, action_record, rnd, cost, bet)

    # 1) opponent's range: every raise keeps the strongest share of hands,
    #    ordered by how strong the hand looked when they raised
    base = initial_range(hand, public)
    opp_range = dict(base)
    for r, facing in opp_raises:
        if facing is None:
            p_raise = np.mean([get_probs(opp_estimates, (r, f))["raise"] for f in (True, False)])
        else:
            p_raise = get_probs(opp_estimates, (r, facing))["raise"]
        frac = float(np.clip(p_raise, MIN_RAISE_RANGE, 1.0))
        opp_range = top_fraction(opp_range, frac, public if r == 1 else None)
    if range_trust < 1.0:
        opp_range = mix(opp_range, base, range_trust)
    eq_now = equity(hand, public, opp_range)

    ev = {}
    # never fold when checking is free (RLCard still lists fold as legal)
    if "fold" in legal and "check" not in legal:
        ev["fold"] = 0.0

    # 2) check: if the opponent has already acted this round our check closes
    #    the betting; otherwise they may bet behind us with their strongest hands.
    if "check" in legal:
        if opp_acted:
            ev["check"] = eq_now * pot
        else:
            p_bet = get_probs(opp_estimates, (rnd, False))["raise"]
            bet_rng = top_fraction(opp_range, p_bet, public)
            chk_rng = subtract(opp_range, bet_rng)
            ev_vs_bet = max(0.0, equity(hand, public, bet_rng) * (pot + 2 * bet) - bet) if bet_rng else 0.0
            ev_vs_chk = equity(hand, public, chk_rng) * pot if chk_rng else 0.0
            ev["check"] = p_bet * ev_vs_bet + (1 - p_bet) * ev_vs_chk

    # 3) call: put in `cost`, then (simplification) showdown
    if "call" in legal:
        ev["call"] = eq_now * (pot + cost) - cost

    # 4) raise: they fold the weakest `fold_rate` of all hands they could hold;
    #    the part of their current range above that line continues.
    p_fold, eq_cont = 0.0, eq_now
    if "raise" in legal:
        fold_rate = get_probs(opp_estimates, (rnd, True))["fold"]
        keep = top_fraction(base, 1 - fold_rate, public)
        stay = {c: keep.get(c, 0.0) / w for c, w in base.items()}   # share of each card that continues
        total = sum(opp_range.values())
        cont_range = {c: w * stay[c] for c, w in opp_range.items() if stay[c] > 1e-12}
        p_fold = 1 - sum(cont_range.values()) / total
        eq_cont = equity(hand, public, cont_range) if cont_range else eq_now
        invest = cost + bet
        # simplification: an opponent re-raise is treated as a call
        ev["raise"] = p_fold * pot + (1 - p_fold) * (eq_cont * (pot + invest + bet) - invest)

    # 5) choose
    actions = list(ev)
    values = np.array([ev[a] for a in actions])
    if temperature > 0:
        p = np.exp((values - values.max()) / temperature)
        p /= p.sum()
        action = np.random.default_rng(rng_seed).choice(actions, p=p)
    else:
        action = actions[int(values.argmax())]

    info = {"equity": round(float(eq_now), 3), "equity_if_called": round(float(eq_cont), 3),
            "opp_fold_to_raise": round(float(p_fold), 3), "opp_raises": opp_raises,
            "cost": cost, "pot": pot, "ev": {a: round(float(v), 2) for a, v in ev.items()}}
    return str(action), info


# ---------------- RLCard agent wrapper ----------------
class AdaptivePolicyAgent:
    """Plugs into RLCard. use_raw=True means step() returns action strings."""
    use_raw = True

    def __init__(self, opponent_model=None, temperature=0.0, range_trust=1.0):
        self.opponent_model = opponent_model
        self.temperature = temperature
        self.range_trust = range_trust

    def _estimates(self):
        return self.opponent_model.get_estimates() if self.opponent_model else None

    def _decide(self, state, temperature):
        return decide(state["raw_obs"], self._estimates(), state.get("action_record"),
                      temperature=temperature, range_trust=self.range_trust)

    def step(self, state):
        return self._decide(state, self.temperature)[0]

    def eval_step(self, state):
        return self._decide(state, 0.0)


# ---------------- demo: same situations, different opponents ----------------
def profile(fold_vs_bet, call_vs_bet, raise_vs_bet, check_no_bet, raise_no_bet):
    ctx = {}
    for r in (0, 1):
        ctx[(r, True)] = {"fold": fold_vs_bet, "call": call_vs_bet, "raise": raise_vs_bet}
        ctx[(r, False)] = {"fold": 0.0, "check": check_no_bet, "raise": raise_no_bet}
    return {"by_context": ctx}


if __name__ == "__main__":
    opponents = {
        "Unknown":      None,
        "Aggressive":   profile(0.05, 0.25, 0.70, 0.30, 0.70),
        "Conservative": profile(0.55, 0.40, 0.05, 0.90, 0.10),
    }
    # current_player is us; records follow RLCard turn order
    situations = {
        "J, preflop, SB called, we can check":
            (dict(hand="SJ", public_card=None, all_chips=[2, 2], my_chips=2, current_player=1,
                  legal_actions=["raise", "fold", "check"]), [(0, "call")]),
        "Q, board K, opponent bet into us":
            (dict(hand="SQ", public_card="HK", all_chips=[8, 4], my_chips=4, current_player=1,
                  legal_actions=["call", "raise", "fold"]), [(0, "raise"), (1, "call"), (0, "raise")]),
        "K, board J, we act first":
            (dict(hand="SK", public_card="HJ", all_chips=[4, 4], my_chips=4, current_player=0,
                  legal_actions=["raise", "fold", "check"]), [(0, "raise"), (1, "call")]),
        "K, board Q, opponent re-raised":
            (dict(hand="SK", public_card="HQ", all_chips=[10, 6], my_chips=6, current_player=1,
                  legal_actions=["call", "fold"]), [(1, "call"), (0, "check"), (1, "raise"), (0, "raise")]),
    }
    for name, (obs, record) in situations.items():
        print(f"\n{name}")
        for opp, est in opponents.items():
            action, info = decide(obs, est, record)
            print(f"  vs {opp:<12} -> {action:<5}  equity={info['equity']:.2f}  EVs={info['ev']}")
