
"""
decision_policy.py - Opponent-aware decision policy for Leduc Hold'em (RLCard).

Inputs:
    - RLCard raw_obs
    - Opponent estimates from the Bayesian opponent model

Output:
    - A legal action: fold, call, check, or raise.

Main components:
    1. Hand equity calculation
    2. Opponent hand-range estimation
    3. Fold-equity estimation
    4. Expected-value calculations
    5. Best-action selection
"""

import numpy as np


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RANKS = {"J": 0, "Q": 1, "K": 2}

DECK = [
    "SJ", "HJ",
    "SQ", "HQ",
    "SK", "HK"
]

RAISE_AMOUNT = {
    0: 2,
    1: 4
}

UNIFORM = {
    "fold": 1 / 3,
    "call": 1 / 3,
    "raise": 1 / 3
}


# --------------------------------------------------
# Hand evaluation
# --------------------------------------------------

def rank(card):
    """Return the numerical rank of a card."""
    return RANKS[card[1]]


def showdown(ours, theirs, public):
    """
    Compare two hands.

    Returns:
        1.0 = We win
        0.5 = Tie
        0.0 = We lose
    """

    our_pair = rank(ours) == rank(public)
    their_pair = rank(theirs) == rank(public)

    if our_pair != their_pair:
        return 1.0 if our_pair else 0.0

    if rank(ours) == rank(theirs):
        return 0.5

    return 1.0 if rank(ours) > rank(theirs) else 0.0


def opp_strength(card, public):
    """
    Estimate how strong an opponent's holding looks.

    A pair is considered stronger than a non-pair.
    """

    if public is not None and rank(card) == rank(public):
        return 10 + rank(card)

    return rank(card)


def initial_range(hand, public):
    """
    Generate all possible opponent cards.

    Cards already visible are excluded.
    """

    seen = {hand}

    if public is not None:
        seen.add(public)

    return {
        card: 1.0
        for card in DECK
        if card not in seen
    }


def top_fraction(rng, frac, public):
    """
    Keep the strongest fraction of the opponent's range.

    This implements the assumption that stronger hands
    are more likely to raise.
    """

    frac = float(np.clip(frac, 0.05, 1.0))

    total = sum(rng.values())

    budget = frac * total
    out = {}

    for card in sorted(
        rng,
        key=lambda c: -opp_strength(c, public)
    ):

        take = min(rng[card], budget)

        if take > 0:
            out[card] = take

        budget -= take

        if budget <= 0:
            break

    return out


def equity(hand, public, rng):
    """
    Calculate our probability of winning against
    a weighted opponent hand range.

    Before the public card is revealed, average
    over all possible public cards.
    """

    num = 0.0
    den = 0.0

    for opponent_card, weight in rng.items():

        if public is not None:

            eq = showdown(
                hand,
                opponent_card,
                public
            )

        else:

            boards = [
                card
                for card in DECK
                if card not in (hand, opponent_card)
            ]

            eq = np.mean([
                showdown(hand, opponent_card, board)
                for board in boards
            ])

        num += weight * eq
        den += weight

    return num / den if den > 0 else 0.5


# --------------------------------------------------
# Opponent model integration
# --------------------------------------------------

def get_probs(estimates, context):
    """
    Retrieve opponent probabilities for a specific context.

    Context:
        (round_number, facing_bet)

    If no estimates are available, use a uniform fallback.
    """

    if not estimates:
        return UNIFORM.copy()

    probabilities = (
        estimates
        .get("by_context", {})
        .get(context)
    )

    if not probabilities:
        return UNIFORM.copy()

    return probabilities


# --------------------------------------------------
# Chip and betting information
# --------------------------------------------------

def get_chip_information(raw_obs):
    """
    Identify our contribution and opponent contribution
    using RLCard's current_player field.

    RLCard stores all_chips in player-index order:
        all_chips[0] -> Player 0's contribution
        all_chips[1] -> Player 1's contribution
    """

    all_chips = raw_obs["all_chips"]
    current_player = raw_obs["current_player"]

    if len(all_chips) != 2:
        raise ValueError(
            "Expected exactly two players in Leduc Hold'em."
        )

    if current_player not in (0, 1):
        raise ValueError("Invalid current_player value.")

    # Identify each player's contribution using player index.
    my_chips = all_chips[current_player]
    opponent_chips = all_chips[1 - current_player]

    # Check that our interpretation agrees with RLCard's field.
    if my_chips != raw_obs["my_chips"]:
        raise ValueError(
            "Chip information does not match RLCard observation."
        )

    pot = sum(all_chips)

    # Amount we need to contribute to match the opponent.
    cost = max(0, opponent_chips - my_chips)

    return {
        "my_chips": my_chips,
        "opponent_chips": opponent_chips,
        "pot": pot,
        "cost": cost
    }


# --------------------------------------------------
# Main decision policy
# --------------------------------------------------

def decide(
    raw_obs,
    opp_estimates=None,
    opp_raised=None,
    temperature=0.0,
    rng_seed=None,
    opp_action_facing_bet=False
):
    """
    Make an opponent-aware decision.

    Parameters
    ----------
    raw_obs:
        RLCard state['raw_obs'] for our player.

    opp_estimates:
        Bayesian opponent estimates.

    opp_raised:
        Whether the opponent raised before our decision.
        If None, inferred from legal actions.

    temperature:
        0 = Select the highest-EV action.
        >0 = Softmax sampling.

    rng_seed:
        Optional seed for softmax sampling.

    opp_action_facing_bet:
        Context in which the opponent made the previous
        raise, when opp_raised is True.

        Defaults to False (an opening raise).

    Returns
    -------
    action, info
    """

    # ----------------------------------------------
    # 1. Read game state
    # ----------------------------------------------

    hand = raw_obs["hand"]

    public = raw_obs.get("public_card")

    legal = [
        str(action)
        for action in raw_obs["legal_actions"]
    ]

    if not legal:
        raise ValueError("No legal actions available.")

    rnd = 0 if public is None else 1

    bet = RAISE_AMOUNT[rnd]

    # ----------------------------------------------
    # 2. Obtain chip information
    # ----------------------------------------------

    chip_info = get_chip_information(raw_obs)

    my_chips = chip_info["my_chips"]
    opponent_chips = chip_info["opponent_chips"]

    pot = chip_info["pot"]
    cost = chip_info["cost"]

    # ----------------------------------------------
    # 3. Determine whether opponent raised
    # ----------------------------------------------

    if opp_raised is None:

        # In RLCard Leduc, call being legal indicates
        # that we are facing a bet.
        opp_raised = "call" in legal

    # ----------------------------------------------
    # 4. Estimate opponent hand range
    # ----------------------------------------------

    opp_range = initial_range(
        hand,
        public
    )

    kept = 1.0

    if opp_raised:

        # Use the context in which the opponent
        # made their previous raise.
        raise_context = (
            rnd,
            opp_action_facing_bet
        )

        raise_freq = get_probs(
            opp_estimates,
            raise_context
        )["raise"]

        kept = float(
            np.clip(raise_freq, 0.15, 1.0)
        )

        opp_range = top_fraction(
            opp_range,
            kept,
            public
        )

    # Calculate our current hand equity.
    eq_now = equity(
        hand,
        public,
        opp_range
    )

    # ----------------------------------------------
    # 5. Estimate opponent fold probability
    # ----------------------------------------------

    # When we raise, the opponent will face a bet.
    response_context = (rnd, True)

    response_probs = get_probs(
        opp_estimates,
        response_context
    )

    fold_rate = response_probs["fold"]

    # Account for the fact that an opponent who
    # already raised is likely to have a stronger range.
    p_fold = (
        max(0.0, kept + fold_rate - 1.0) / kept
    )

    p_fold = float(
        np.clip(p_fold, 0.0, 1.0)
    )

    # Estimate the opponent's continuing range.
    cont_range = top_fraction(
        opp_range,
        1.0 - p_fold,
        public
    )

    eq_cont = equity(
        hand,
        public,
        cont_range
    )

    # ----------------------------------------------
    # 6. Calculate expected value of each action
    # ----------------------------------------------

    ev = {}

    # Folding
    if "fold" in legal and "check" not in legal:
        ev["fold"] = 0.0

    # Checking
    if "check" in legal:

        ev["check"] = eq_now * pot

    # Calling
    if "call" in legal:

        ev["call"] = (
            eq_now * (pot + cost)
            - cost
        )

    # Raising
    if "raise" in legal:

        invest = cost + bet

        ev["raise"] = (
            p_fold * pot
            +
            (1.0 - p_fold)
            *
            (
                eq_cont * (pot + invest + bet)
                - invest
            )
        )

    if not ev:
        raise ValueError(
            "Could not calculate EV for any legal action."
        )

    # ----------------------------------------------
    # 7. Select the best action
    # ----------------------------------------------

    actions = list(ev.keys())

    values = np.array([
        ev[action]
        for action in actions
    ])

    if temperature > 0:

        probabilities = np.exp(
            (values - values.max()) / temperature
        )

        probabilities /= probabilities.sum()

        action = np.random.default_rng(
            rng_seed
        ).choice(
            actions,
            p=probabilities
        )

    else:

        action = actions[
            int(values.argmax())
        ]

    # ----------------------------------------------
    # 8. Return decision information
    # ----------------------------------------------

    info = {
        "round": rnd,

        "hand": hand,

        "public_card": public,

        "my_chips": my_chips,

        "opponent_chips": opponent_chips,

        "pot": pot,

        "call_cost": cost,

        "opp_raised": bool(opp_raised),

        "opponent_raise_frequency": round(
            float(raise_freq), 3
        ) if opp_raised else None,

        "opponent_fold_probability": round(
            float(fold_rate), 3
        ),

        "opp_fold_to_raise": round(
            float(p_fold), 3
        ),

        "equity": round(
            float(eq_now), 3
        ),

        "equity_if_called": round(
            float(eq_cont), 3
        ),

        "ev": {
            action: round(float(value), 3)
            for action, value in ev.items()
        },

        "selected_action": str(action)
    }

    return str(action), info


# --------------------------------------------------
# RLCard agent wrapper
# --------------------------------------------------

class AdaptivePolicyAgent:
    """
    RLCard-compatible adaptive policy agent.

    The Bayesian model is queried before each decision.
    """

    use_raw = True

    def __init__(
        self,
        opponent_model=None,
        temperature=0.0
    ):

        self.opponent_model = opponent_model
        self.temperature = temperature

    def _estimates(self):

        if self.opponent_model is None:
            return None

        return self.opponent_model.get_estimates()

    def step(self, state):

        action, info = decide(
            state["raw_obs"],
            self._estimates(),
            temperature=self.temperature
        )

        return action

    def eval_step(self, state):

        action, info = decide(
            state["raw_obs"],
            self._estimates(),
            temperature=self.temperature
        )

        return action, info


# --------------------------------------------------
# Test helper: create opponent profiles
# --------------------------------------------------

def profile(
    fold_vs_bet,
    call_vs_bet,
    raise_vs_bet,
    call_no_bet,
    raise_no_bet
):

    ctx = {}

    for r in (0, 1):

        ctx[(r, True)] = {
            "fold": fold_vs_bet,
            "call": call_vs_bet,
            "raise": raise_vs_bet
        }

        ctx[(r, False)] = {
            "fold": 0.0,
            "call": call_no_bet,
            "raise": raise_no_bet
        }

    return {
        "by_context": ctx
    }


# --------------------------------------------------
# Basic decision policy demonstration
# --------------------------------------------------

if __name__ == "__main__":

    opponents = {

        "Unknown": None,

        "Aggressive": profile(
            0.05, 0.25, 0.70,
            0.30, 0.70
        ),

        "Conservative": profile(
            0.55, 0.40, 0.05,
            0.90, 0.10
        )
    }

    situations = {

        "J, preflop, no bet to face": {
            "hand": "SJ",
            "public_card": None,
            "all_chips": [2, 2],
            "my_chips": 2,
            "current_player": 0,
            "legal_actions": [
                "raise", "fold", "check"
            ]
        },

        "Q, board K, facing a raise": {
            "hand": "SQ",
            "public_card": "HK",
            "all_chips": [4, 8],
            "my_chips": 4,
            "current_player": 0,
            "legal_actions": [
                "call", "raise", "fold"
            ]
        },

        "K, board J, no bet to face": {
            "hand": "SK",
            "public_card": "HJ",
            "all_chips": [4, 4],
            "my_chips": 4,
            "current_player": 0,
            "legal_actions": [
                "raise", "fold", "check"
            ]
        },

        "K, board Q, facing a raise": {
            "hand": "SK",
            "public_card": "HQ",
            "all_chips": [6, 10],
            "my_chips": 6,
            "current_player": 0,
            "legal_actions": [
                "call", "fold"
            ]
        }
    }

    for name, obs in situations.items():

        print(f"\n{name}")

        for opponent_name, estimates in opponents.items():

            action, info = decide(
                obs,
                estimates
            )

            print(
                f"  vs {opponent_name:<12} "
                f"-> {action:<5} "
                f"equity={info['equity']:.2f} "
                f"EVs={info['ev']}"
            )
