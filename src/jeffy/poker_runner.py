"""Poker runner for live classifier demo.

Runs a 4-player Texas Hold'em table, extracts features each decision,
runs the Jeffy classifier, and yields game state for the WebSocket stream.
"""

import itertools
import random
import time
from collections import Counter

import numpy as np

RANKS = list(range(2, 15))  # 2-14 (14=Ace)
SUITS = ["s", "h", "d", "c"]
RANK_NAMES = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
              9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}
SUIT_SYMBOLS = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}

ACTIONS = ["fold", "check", "call", "raise", "all_in"]
N_FEATURES = 18

PLAYER_NAMES = ["Alice", "Bob", "Carol", "Dave"]

SMALL_BLIND = 5
BIG_BLIND = 10
STARTING_STACK = 1000


def card_str(card):
    return RANK_NAMES[card[0]] + SUIT_SYMBOLS[card[1]]


def card_code(card):
    return RANK_NAMES[card[0]] + card[1]


# --- Hand evaluation ---

HAND_RANKS = {
    "high_card": 0, "pair": 1, "two_pair": 2, "three_kind": 3,
    "straight": 4, "flush": 5, "full_house": 6, "four_kind": 7,
    "straight_flush": 8, "royal_flush": 9,
}


def _best_five(cards):
    """Evaluate the best 5-card hand from a list of cards.
    Returns (rank_value, tiebreaker_tuple, hand_name)."""
    best = None
    for combo in itertools.combinations(cards, 5):
        val = _eval_five(combo)
        if best is None or val > best:
            best = val
    return best


def _eval_five(cards):
    ranks = sorted([c[0] for c in cards], reverse=True)
    suits = [c[1] for c in cards]
    rc = Counter(ranks)
    is_flush = len(set(suits)) == 1

    sorted_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    high = 0
    if len(sorted_ranks) == 5:
        if sorted_ranks[0] - sorted_ranks[4] == 4:
            is_straight = True
            high = sorted_ranks[0]
        elif sorted_ranks == [14, 5, 4, 3, 2]:
            is_straight = True
            high = 5  # wheel

    freqs = sorted(rc.values(), reverse=True)

    if is_straight and is_flush:
        if high == 14 and min(ranks) == 10:
            return (9, (14,), "royal_flush")
        return (8, (high,), "straight_flush")
    if freqs == [4, 1]:
        quad = [r for r, c in rc.items() if c == 4][0]
        kick = [r for r, c in rc.items() if c == 1][0]
        return (7, (quad, kick), "four_kind")
    if freqs == [3, 2]:
        trip = [r for r, c in rc.items() if c == 3][0]
        pair = [r for r, c in rc.items() if c == 2][0]
        return (6, (trip, pair), "full_house")
    if is_flush:
        return (5, tuple(ranks), "flush")
    if is_straight:
        return (4, (high,), "straight")
    if freqs == [3, 1, 1]:
        trip = [r for r, c in rc.items() if c == 3][0]
        kicks = sorted([r for r, c in rc.items() if c == 1], reverse=True)
        return (3, (trip,) + tuple(kicks), "three_kind")
    if freqs == [2, 2, 1]:
        pairs = sorted([r for r, c in rc.items() if c == 2], reverse=True)
        kick = [r for r, c in rc.items() if c == 1][0]
        return (2, tuple(pairs) + (kick,), "two_pair")
    if freqs == [2, 1, 1, 1]:
        pair = [r for r, c in rc.items() if c == 2][0]
        kicks = sorted([r for r, c in rc.items() if c == 1], reverse=True)
        return (1, (pair,) + tuple(kicks), "pair")
    return (0, tuple(ranks), "high_card")


def hand_strength_score(hole, community):
    """0-1 score of current hand strength."""
    if not community:
        return _preflop_strength(hole)
    cards = list(hole) + list(community)
    val = _best_five(cards)
    rank_val = val[0]
    return min(1.0, (rank_val + 0.5) / 9.5)


def hand_name(hole, community):
    cards = list(hole) + list(community)
    if len(cards) < 5:
        return "preflop"
    val = _best_five(cards)
    return val[2]


def _preflop_strength(hole):
    r1, r2 = sorted([hole[0][0], hole[1][0]], reverse=True)
    suited = hole[0][1] == hole[1][1]
    if r1 == r2:
        base = 0.5 + (r1 - 2) / 24.0
    else:
        gap = r1 - r2
        base = 0.2 + (r1 + r2 - 4) / 52.0 - gap * 0.02
        if suited:
            base += 0.04
    return min(1.0, max(0.0, base))


# --- Feature extraction ---

def extract_features(player_idx, game_state):
    """Extract features for a player's decision."""
    p = game_state["players"][player_idx]
    hole = p["hole"]
    community = game_state["community"]
    pot = game_state["pot"]
    current_bet = game_state["current_bet"]
    player_bet = p["bet"]
    to_call = current_bet - player_bet
    big_blind = game_state["big_blind"]

    active_opponents = sum(
        1 for i, op in enumerate(game_state["players"])
        if i != player_idx and op["active"] and not op["folded"]
    )

    n_raises = game_state.get("n_raises_this_round", 0)
    round_idx = ["preflop", "flop", "turn", "river"].index(game_state["round"])

    strength = hand_strength_score(hole, community)
    hname = hand_name(hole, community)
    made_rank = HAND_RANKS.get(hname, 0) / 9.0

    r1, r2 = sorted([hole[0][0], hole[1][0]], reverse=True)
    is_pair = 1.0 if r1 == r2 else 0.0
    is_suited = 1.0 if hole[0][1] == hole[1][1] else 0.0
    high_card = r1 / 14.0
    card_gap = (r1 - r2) / 12.0

    pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0
    stack_to_pot = p["stack"] / pot if pot > 0 else 10.0
    stack_to_blind = p["stack"] / big_blind if big_blind > 0 else 100.0

    position = player_idx / 3.0  # 0=first, 1=last

    features = np.array([
        strength,
        made_rank,
        round_idx / 3.0,
        pot_odds,
        stack_to_pot,
        min(stack_to_blind, 200) / 200.0,
        position,
        active_opponents / 3.0,
        min(n_raises, 4) / 4.0,
        min(to_call, 200) / 200.0,
        is_pair,
        is_suited,
        high_card,
        card_gap,
        1.0 if round_idx == 0 else 0.0,
        min(pot, 500) / 500.0,
        min(p["stack"], 2000) / 2000.0,
        p["bet"] / max(current_bet, 1),
    ], dtype=np.float64)
    return features


# --- Rule-based strategy for training data ---

def rule_based_decision(features, rng=None):
    """Generate a labeled decision from features using poker heuristics."""
    if rng is None:
        rng = random.Random()

    strength = features[0]
    made_rank = features[1]
    round_f = features[2]
    pot_odds = features[3]
    stack_to_pot = features[4]
    n_raises = features[8] * 4
    to_call_f = features[9] * 200
    is_preflop = features[14]

    noise = rng.gauss(0, 0.05)
    adj_strength = strength + noise

    if is_preflop > 0.5:
        if adj_strength > 0.7:
            return "raise" if rng.random() < 0.7 else "call"
        elif adj_strength > 0.45:
            if to_call_f < 20:
                return "call"
            elif adj_strength > 0.55:
                return "call" if rng.random() < 0.6 else "raise"
            else:
                return "fold" if rng.random() < 0.5 else "call"
        else:
            if to_call_f < 10 and rng.random() < 0.3:
                return "call"
            return "fold"
    else:
        if adj_strength > 0.75 or made_rank > 0.5:
            if rng.random() < 0.15 and stack_to_pot < 3:
                return "all_in"
            return "raise" if rng.random() < 0.65 else "call"
        elif adj_strength > 0.5:
            if pot_odds < 0.3:
                return "call"
            elif n_raises >= 2:
                return "fold" if rng.random() < 0.6 else "call"
            else:
                return "call" if rng.random() < 0.7 else "raise"
        elif adj_strength > 0.3:
            if to_call_f < 1:
                return "check"
            elif pot_odds < 0.2:
                return "call" if rng.random() < 0.5 else "fold"
            else:
                return "fold"
        else:
            if to_call_f < 1:
                return "check"
            return "fold"


# --- Texas Hold'em game engine ---

class PokerGame:
    def __init__(self, n_players=4, rng_seed=None):
        self.n_players = n_players
        self.rng = random.Random(rng_seed)
        self.stacks = [STARTING_STACK] * n_players
        self.dealer = 0
        self.hand_number = 0

    def new_hand(self):
        self.hand_number += 1
        deck = [(r, s) for r in RANKS for s in SUITS]
        self.rng.shuffle(deck)

        holes = [tuple(deck[i * 2:(i + 1) * 2]) for i in range(self.n_players)]
        community_full = deck[self.n_players * 2:self.n_players * 2 + 5]

        self.dealer = (self.dealer + 1) % self.n_players

        state = {
            "hand_number": self.hand_number,
            "dealer": self.dealer,
            "big_blind": BIG_BLIND,
            "small_blind": SMALL_BLIND,
            "pot": 0,
            "current_bet": 0,
            "community": [],
            "community_full": community_full,
            "round": "preflop",
            "n_raises_this_round": 0,
            "players": [],
        }

        for i in range(self.n_players):
            state["players"].append({
                "name": PLAYER_NAMES[i],
                "hole": holes[i],
                "stack": self.stacks[i],
                "bet": 0,
                "folded": False,
                "active": self.stacks[i] > 0,
                "last_action": None,
            })

        # Post blinds
        sb_idx = (self.dealer + 1) % self.n_players
        bb_idx = (self.dealer + 2) % self.n_players
        sb_amt = min(SMALL_BLIND, state["players"][sb_idx]["stack"])
        bb_amt = min(BIG_BLIND, state["players"][bb_idx]["stack"])
        state["players"][sb_idx]["stack"] -= sb_amt
        state["players"][sb_idx]["bet"] = sb_amt
        state["players"][bb_idx]["stack"] -= bb_amt
        state["players"][bb_idx]["bet"] = bb_amt
        state["pot"] = sb_amt + bb_amt
        state["current_bet"] = bb_amt

        return state

    def apply_action(self, state, player_idx, action):
        p = state["players"][player_idx]
        to_call = state["current_bet"] - p["bet"]

        if action == "fold":
            p["folded"] = True
            p["last_action"] = "fold"
        elif action == "check":
            p["last_action"] = "check"
        elif action == "call":
            amt = min(to_call, p["stack"])
            p["stack"] -= amt
            p["bet"] += amt
            state["pot"] += amt
            p["last_action"] = "call"
        elif action == "raise":
            call_amt = min(to_call, p["stack"])
            raise_amt = min(BIG_BLIND * 2, p["stack"] - call_amt)
            total = call_amt + raise_amt
            p["stack"] -= total
            p["bet"] += total
            state["pot"] += total
            state["current_bet"] = p["bet"]
            state["n_raises_this_round"] += 1
            p["last_action"] = "raise"
        elif action == "all_in":
            amt = p["stack"]
            p["bet"] += amt
            state["pot"] += amt
            if p["bet"] > state["current_bet"]:
                state["current_bet"] = p["bet"]
                state["n_raises_this_round"] += 1
            p["stack"] = 0
            p["last_action"] = "all_in"

        return state

    def sanitize_action(self, state, player_idx, action):
        """Ensure action is legal given game state."""
        p = state["players"][player_idx]
        to_call = state["current_bet"] - p["bet"]

        if p["stack"] <= 0:
            return "check" if to_call <= 0 else "fold"

        if action == "check" and to_call > 0:
            return "fold"
        if action == "call" and to_call <= 0:
            return "check"
        if action == "raise" and p["stack"] <= to_call:
            return "all_in" if to_call > 0 else "check"

        return action

    def advance_round(self, state):
        rounds = ["preflop", "flop", "turn", "river"]
        idx = rounds.index(state["round"])
        if idx >= 3:
            return False

        state["round"] = rounds[idx + 1]
        state["n_raises_this_round"] = 0
        for p in state["players"]:
            p["bet"] = 0
        state["current_bet"] = 0

        if state["round"] == "flop":
            state["community"] = list(state["community_full"][:3])
        elif state["round"] == "turn":
            state["community"] = list(state["community_full"][:4])
        elif state["round"] == "river":
            state["community"] = list(state["community_full"][:5])

        return True

    def showdown(self, state):
        """Determine winner and award pot."""
        active = [
            (i, p) for i, p in enumerate(state["players"])
            if not p["folded"] and p["active"]
        ]
        if len(active) == 1:
            winner_idx = active[0][0]
            state["players"][winner_idx]["stack"] += state["pot"]
            return winner_idx, "last_standing", None

        best_val = None
        winner_idx = None
        winner_hand = None
        for i, p in active:
            cards = list(p["hole"]) + list(state["community"])
            if len(cards) >= 5:
                val = _best_five(cards)
            else:
                val = (0, tuple(sorted([c[0] for c in p["hole"]], reverse=True)), "preflop")
            if best_val is None or val > best_val:
                best_val = val
                winner_idx = i
                winner_hand = val[2]

        state["players"][winner_idx]["stack"] += state["pot"]
        return winner_idx, winner_hand, best_val

    def sync_stacks(self, state):
        for i, p in enumerate(state["players"]):
            self.stacks[i] = p["stack"]

    def betting_order(self, state):
        """Return player indices in betting order for current round."""
        if state["round"] == "preflop":
            start = (self.dealer + 3) % self.n_players
        else:
            start = (self.dealer + 1) % self.n_players

        order = []
        for i in range(self.n_players):
            idx = (start + i) % self.n_players
            p = state["players"][idx]
            if p["active"] and not p["folded"] and p["stack"] > 0:
                order.append(idx)
        return order


# --- Training data generation ---

def generate_training_data(n_hands=2000, seed=42):
    """Generate (features, label) pairs by simulating hands."""
    game = PokerGame(rng_seed=seed)
    rng = random.Random(seed)
    all_features = []
    all_labels = []

    for _ in range(n_hands):
        # Reset busted players
        for i in range(4):
            if game.stacks[i] < BIG_BLIND:
                game.stacks[i] = STARTING_STACK

        state = game.new_hand()

        for round_name in ["preflop", "flop", "turn", "river"]:
            if state["round"] != round_name:
                if round_name != "preflop":
                    if not game.advance_round(state):
                        break
                    if state["round"] != round_name:
                        break

            order = game.betting_order(state)
            active_count = sum(
                1 for p in state["players"] if not p["folded"] and p["active"]
            )
            if active_count <= 1:
                break

            for pidx in order:
                active_count = sum(
                    1 for p in state["players"] if not p["folded"] and p["active"]
                )
                if active_count <= 1:
                    break

                feats = extract_features(pidx, state)
                decision = rule_based_decision(feats, rng)
                decision = game.sanitize_action(state, pidx, decision)

                all_features.append(feats)
                all_labels.append(decision)

                game.apply_action(state, pidx, decision)

            if round_name != "river":
                if not game.advance_round(state):
                    break

        game.showdown(state)
        game.sync_stacks(state)

    return np.array(all_features), all_labels


# --- Live session for WebSocket streaming ---

class PokerSession:
    def __init__(self, engine, task_id="poker_decision"):
        self.engine = engine
        self.task_id = task_id
        self.game = PokerGame(rng_seed=int(time.time()))
        self.state = None
        self.hand_history = []
        self.current_decisions = []
        self._round_idx = 0
        self._action_queue = []
        self._phase = "deal"
        self._wait_until = 0
        self._hand_count = 0

    def start(self):
        self._start_new_hand()

    def _start_new_hand(self):
        for i in range(4):
            if self.game.stacks[i] < BIG_BLIND * 2:
                self.game.stacks[i] = STARTING_STACK

        self.state = self.game.new_hand()
        self._round_idx = 0
        self._phase = "deal"
        self._action_queue = []
        self.current_decisions = []
        self._hand_count += 1

    def tick(self):
        """Advance game by one step. Returns game state dict or None."""
        if self.state is None:
            return None

        now = time.time()
        if now < self._wait_until:
            return self._build_output("waiting")

        active_count = sum(
            1 for p in self.state["players"]
            if not p["folded"] and p["active"]
        )

        if self._phase == "deal":
            self._phase = "betting"
            self._action_queue = list(self.game.betting_order(self.state))
            return self._build_output("deal")

        if self._phase == "betting":
            if not self._action_queue or active_count <= 1:
                return self._advance_or_showdown()

            pidx = self._action_queue.pop(0)
            p = self.state["players"][pidx]
            if p["folded"] or not p["active"] or p["stack"] <= 0:
                return self._build_output("skip")

            feats = extract_features(pidx, self.state)
            result = self.engine.predict_features(self.task_id, feats.tolist())

            action = result["label"]
            action = self.game.sanitize_action(self.state, pidx, action)
            self.game.apply_action(self.state, pidx, action)

            decision_info = {
                "player": pidx,
                "name": PLAYER_NAMES[pidx],
                "action": action,
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "classify_ms": result.get("classifier_ms", 0),
            }
            self.current_decisions.append(decision_info)

            self._wait_until = now + 0.3
            return self._build_output("action", decision_info)

        if self._phase == "showdown":
            winner_idx, win_reason, _ = self.game.showdown(self.state)
            self.game.sync_stacks(self.state)
            winner_name = PLAYER_NAMES[winner_idx]

            self.hand_history.append({
                "hand": self._hand_count,
                "winner": winner_name,
                "reason": win_reason,
                "pot": self.state["pot"],
            })
            if len(self.hand_history) > 20:
                self.hand_history.pop(0)

            self._phase = "hand_over"
            self._wait_until = now + 2.0
            return self._build_output("showdown", {
                "winner": winner_idx,
                "winner_name": winner_name,
                "win_reason": win_reason,
                "pot": self.state["pot"],
            })

        if self._phase == "hand_over":
            self._start_new_hand()
            self._wait_until = now + 1.0
            return self._build_output("new_hand")

        return None

    def _advance_or_showdown(self):
        active_count = sum(
            1 for p in self.state["players"]
            if not p["folded"] and p["active"]
        )

        if active_count <= 1 or self.state["round"] == "river":
            self._phase = "showdown"
            if self.state["round"] != "river" and active_count > 1:
                while self.game.advance_round(self.state):
                    if self.state["round"] == "river":
                        break
            return self._build_output("to_showdown")

        self.game.advance_round(self.state)
        self._action_queue = list(self.game.betting_order(self.state))
        self._wait_until = time.time() + 0.8
        return self._build_output("new_round")

    def _build_output(self, event, extra=None):
        players = []
        for i, p in enumerate(self.state["players"]):
            players.append({
                "name": p["name"],
                "hole": [card_code(c) for c in p["hole"]] if p["active"] else [],
                "hole_display": [card_str(c) for c in p["hole"]] if p["active"] else [],
                "stack": p["stack"],
                "bet": p["bet"],
                "folded": p["folded"],
                "active": p["active"],
                "last_action": p["last_action"],
            })

        community_display = [card_str(c) for c in self.state["community"]]

        out = {
            "event": event,
            "hand_number": self._hand_count,
            "round": self.state["round"],
            "pot": self.state["pot"],
            "community": [card_code(c) for c in self.state["community"]],
            "community_display": community_display,
            "dealer": self.state["dealer"],
            "players": players,
            "decisions": self.current_decisions[-8:],
            "hand_history": self.hand_history[-10:],
        }
        if extra:
            out["detail"] = extra
        return out

    def close(self):
        pass
