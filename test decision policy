"""
test_decision_policy.py - checks decision_policy.py against a re-implementation
of RLCard's Leduc betting rules (rlcard/games/leducholdem/round.py, game.py).

Run: python test_decision_policy.py
Once RLCard is installed, also run check_against_rlcard() (bottom of file),
which compares against the real environment.
"""
import itertools
from decision_policy import (decide, label_actions, opponent_history, get_probs,
                             RAISE_AMOUNT, DECK, profile)


# ---- mini simulator of RLCard Leduc betting ----
def legal_actions(raised, pointer, n_raises):
    acts = ["call", "raise", "fold", "check"]
    if n_raises >= 2:
        acts.remove("raise")
    if raised[pointer] < max(raised):
        acts.remove("check")
    if raised[pointer] == max(raised):
        acts.remove("call")
    return acts


def all_decision_points(sb):
    """Yield every decision point of a hand: (round, pointer, chips, legal, record)."""
    def rec(rnd, pointer, chips, raised, n_raises, counter, record):
        legal = legal_actions(raised, pointer, n_raises)
        yield rnd, pointer, list(chips), legal, list(record)
        for a in legal:
            if a == "fold":
                continue
            c, r, n, k = list(chips), list(raised), n_raises, counter
            if a == "call":
                diff = max(r) - r[pointer]; r[pointer] += diff; c[pointer] += diff; k += 1
            elif a == "check":
                k += 1
            else:  # raise
                diff = max(r) - r[pointer] + RAISE_AMOUNT[rnd]
                r[pointer] += diff; c[pointer] += diff; n += 1; k = 1
            nxt = 1 - pointer
            new_record = record + [(pointer, a)]
            if k >= 2:                       # round over
                if rnd == 1:
                    continue                 # showdown, no more decisions
                yield from rec(1, nxt, c, [0, 0], 0, 0, new_record)
            else:
                yield from rec(rnd, nxt, c, r, n, k, new_record)

    chips = [0, 0]
    chips[sb], chips[1 - sb] = 1, 2
    yield from rec(0, sb, chips, list(chips), 0, 0, [])


def test_labels_rounds_and_raise_detection():
    n = 0
    for sb in (0, 1):
        for rnd, me, chips, legal, record in all_decision_points(sb):
            n += 1
            labelled = label_actions(record + [(me, "check")])   # dummy next action
            _, _, lab_rnd, lab_facing = labelled[-1]
            assert lab_rnd == rnd, (record, lab_rnd, rnd)
            assert lab_facing == ("call" in legal), (record, legal)

            # chip-based fallback must agree with the action record
            obs = dict(all_chips=chips, my_chips=chips[me], current_player=me)
            pot, my = sum(chips), chips[me]
            cost = max(0, (pot - my) - my)
            exact, acted = opponent_history(obs, record, rnd, cost, RAISE_AMOUNT[rnd])
            approx, _ = opponent_history(obs, None, rnd, cost, RAISE_AMOUNT[rnd])
            raised_now = any(r == rnd for r, _ in exact)
            assert raised_now == bool(approx), (record, chips)
            # check is only legal when not behind, and then our check closes the
            # round exactly when the opponent already acted this round
            if "check" in legal:
                assert cost == 0
    print(f"labels / raise detection: {n} decision points OK")


def test_policy_returns_legal_action_everywhere():
    ests = [None, profile(0.05, 0.25, 0.70, 0.30, 0.70), profile(0.55, 0.40, 0.05, 0.90, 0.10),
            {"overall": {"fold": 0.2, "check": 0.3, "call": 0.3, "raise": 0.2}}]
    n = 0
    for sb in (0, 1):
        for rnd, me, chips, legal, record in all_decision_points(sb):
            for hand, opp, pub in itertools.permutations(DECK, 3):
                public = pub if rnd == 1 else None
                obs = dict(hand=hand, public_card=public, all_chips=chips, my_chips=chips[me],
                           current_player=me, legal_actions=legal)
                for est in ests:
                    action, info = decide(obs, est, record)
                    assert action in legal, (obs, action)
                    if "check" in legal:
                        assert action != "fold"
                    n += 1
    print(f"policy legality: {n} decisions OK")


def test_ev_accounting():
    # Calling as small blind preflop: pot 3, cost 1, no raise yet -> EV = eq * 4 - 1
    obs = dict(hand="SK", public_card=None, all_chips=[2, 1], my_chips=1, current_player=1,
               legal_actions=["call", "raise", "fold"])
    _, info = decide(obs, None, [])
    assert info["opp_raises"] == [] and info["cost"] == 1
    assert abs(info["ev"]["call"] - (info["equity"] * 4 - 1)) < 0.02, info
    # Raise EV when opponent always folds = current pot
    est = {"by_context": {(1, True): {"fold": 1.0, "call": 0.0, "raise": 0.0}}}
    obs = dict(hand="SJ", public_card="HQ", all_chips=[4, 4], my_chips=4, current_player=0,
               legal_actions=["raise", "fold", "check"])
    _, info = decide(obs, est, [(0, "call"), (1, "check")])
    assert info["ev"]["raise"] == 8.0, info
    # get_probs merges check+call and renormalises
    p = get_probs({"by_context": {(0, False): {"check": 0.6, "raise": 0.2, "fold": 0.2}}}, (0, False))
    assert abs(p["passive"] - 0.6) < 1e-9
    print("EV accounting OK")


def check_against_rlcard(n_hands=2000):
    """Requires `pip install rlcard`. Verifies chip/label logic on real games."""
    import rlcard
    from rlcard.agents import RandomAgent
    from decision_policy import AdaptivePolicyAgent
    env = rlcard.make("leduc-holdem")
    checked = []

    class Probe(AdaptivePolicyAgent):
        def _decide(self, state, temperature):
            obs, rec = state["raw_obs"], state["action_record"]
            rnd = 0 if obs["public_card"] is None else 1
            labelled = label_actions(rec + [(obs["current_player"], "check")])
            assert labelled[-1][2] == rnd, (rec, obs)
            assert labelled[-1][3] == ("call" in obs["legal_actions"]), (rec, obs)
            checked.append(1)
            return super()._decide(state, temperature)

    env.set_agents([Probe(), RandomAgent(num_actions=env.num_actions)])
    for _ in range(n_hands):
        env.run(is_training=False)
    print(f"RLCard check: {len(checked)} real decisions OK")


if __name__ == "__main__":
    test_labels_rounds_and_raise_detection()
    test_policy_returns_legal_action_everywhere()
    test_ev_accounting()
    try:
        check_against_rlcard()
    except ImportError:
        print("rlcard not installed - skipped the real-environment check")
