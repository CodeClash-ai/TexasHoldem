#!/usr/bin/env python3
"""
Texas Hold'em Starter Bot for CodeClash

This bot implements a basic TAG (Tight-Aggressive) strategy:
- Play strong starting hands
- Bet/raise with strong hands
- Fold weak hands to aggression
- Consider position and pot odds
"""

from collections import Counter
from itertools import combinations


# Hand strength rankings for preflop play
# Premium hands: Always raise
PREMIUM_HANDS = {
    "AA",
    "KK",
    "QQ",
    "JJ",
    "AKs",
    "AKo",
}

# Strong hands: Raise or call raises
STRONG_HANDS = {
    "TT",
    "99",
    "AQs",
    "AQo",
    "AJs",
    "KQs",
}

# Playable hands: Call or raise in position
PLAYABLE_HANDS = {
    "88",
    "77",
    "66",
    "ATs",
    "ATo",
    "KJs",
    "KTs",
    "QJs",
    "JTs",
    "A9s",
    "A8s",
    "A7s",
    "A6s",
    "A5s",
    "A4s",
    "A3s",
    "A2s",
    "KQo",
    "QTs",
    "T9s",
    "98s",
    "87s",
    "76s",
    "65s",
}


def get_hand_category(hole_cards: list[str]) -> str:
    """Convert hole cards to hand notation (e.g., 'AKs' for suited, 'AKo' for offsuit)."""
    RANK_ORDER = "23456789TJQKA"

    r1, r2 = hole_cards[0][0], hole_cards[1][0]
    s1, s2 = hole_cards[0][1], hole_cards[1][1]

    suited = "s" if s1 == s2 else "o"

    # Order ranks by strength
    r1_val = RANK_ORDER.index(r1)
    r2_val = RANK_ORDER.index(r2)

    if r1_val < r2_val:
        r1, r2 = r2, r1

    if r1 == r2:
        return f"{r1}{r2}"  # Pocket pair
    return f"{r1}{r2}{suited}"


def evaluate_hand_strength(
    hole_cards: list[str], community_cards: list[str]
) -> tuple[int, list[int]]:
    """
    Evaluate hand strength. Returns (hand_rank, tiebreaker).
    Hand ranks: 0=high card, 1=pair, 2=two pair, 3=trips, 4=straight,
                5=flush, 6=full house, 7=quads, 8=straight flush, 9=royal flush
    """
    RANKS = "23456789TJQKA"

    all_cards = hole_cards + community_cards
    if len(all_cards) < 5:
        # Preflop - just use high card
        vals = sorted([RANKS.index(c[0]) for c in hole_cards], reverse=True)
        return 0, vals

    best_rank = -1
    best_tie: list[int] = []

    for combo in combinations(all_cards, 5):
        rank, tie = _eval_five(list(combo))
        if rank > best_rank or (rank == best_rank and tie > best_tie):
            best_rank = rank
            best_tie = tie

    return best_rank, best_tie


def _eval_five(cards: list[str]) -> tuple[int, list[int]]:
    """Evaluate exactly 5 cards."""
    RANKS = "23456789TJQKA"

    ranks = sorted([RANKS.index(c[0]) for c in cards], reverse=True)
    suits = [c[1] for c in cards]
    rank_counts = Counter(ranks)
    is_flush = len(set(suits)) == 1

    sorted_ranks = sorted(set(ranks))
    is_straight = False
    straight_high = 0

    if len(sorted_ranks) == 5:
        if sorted_ranks[-1] - sorted_ranks[0] == 4:
            is_straight = True
            straight_high = sorted_ranks[-1]
        elif sorted_ranks == [0, 1, 2, 3, 12]:  # Wheel
            is_straight = True
            straight_high = 3

    counts = sorted(rank_counts.values(), reverse=True)

    if is_straight and is_flush:
        return (9 if straight_high == 12 else 8), [straight_high]
    if counts == [4, 1]:
        quad = [r for r, c in rank_counts.items() if c == 4][0]
        kick = [r for r, c in rank_counts.items() if c == 1][0]
        return 7, [quad, kick]
    if counts == [3, 2]:
        trip = [r for r, c in rank_counts.items() if c == 3][0]
        pair = [r for r, c in rank_counts.items() if c == 2][0]
        return 6, [trip, pair]
    if is_flush:
        return 5, ranks
    if is_straight:
        return 4, [straight_high]
    if counts == [3, 1, 1]:
        trip = [r for r, c in rank_counts.items() if c == 3][0]
        kicks = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return 3, [trip] + kicks
    if counts == [2, 2, 1]:
        pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
        kick = [r for r, c in rank_counts.items() if c == 1][0]
        return 2, pairs + [kick]
    if counts == [2, 1, 1, 1]:
        pair = [r for r, c in rank_counts.items() if c == 2][0]
        kicks = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return 1, [pair] + kicks

    return 0, ranks


def get_move(state) -> str:
    """
    Main decision function called by the engine.

    Args:
        state: GameState object with:
            - hole_cards: Your 2 hole cards
            - community_cards: Current community cards (0-5)
            - pot: Total pot size
            - current_bet: Current bet to call
            - player_stack: Your remaining chips
            - opponent_stack: Opponent's remaining chips
            - player_bet: Amount you've bet this round
            - opponent_bet: Amount opponent has bet this round
            - position: 'button' or 'big_blind'
            - round_name: 'preflop', 'flop', 'turn', 'river'
            - min_raise: Minimum raise amount
            - is_first_action: True if first to act this betting round

    Returns:
        Action string: 'fold', 'call', 'check', 'raise <amount>', or 'all_in'
    """
    hole_cards = state.hole_cards
    community_cards = state.community_cards
    pot = state.pot
    current_bet = state.current_bet
    player_bet = state.player_bet
    player_stack = state.player_stack
    position = state.position
    round_name = state.round_name
    min_raise = state.min_raise

    to_call = current_bet - player_bet

    # Preflop strategy
    if round_name == "preflop":
        return preflop_strategy(
            hole_cards, position, to_call, pot, player_stack, min_raise, current_bet
        )

    # Postflop strategy
    return postflop_strategy(
        hole_cards,
        community_cards,
        to_call,
        pot,
        player_stack,
        min_raise,
        current_bet,
        round_name,
    )


def preflop_strategy(
    hole_cards, position, to_call, pot, stack, min_raise, current_bet
) -> str:
    """Preflop betting strategy based on hand strength and position."""
    hand = get_hand_category(hole_cards)
    in_position = position == "button"

    # Premium hands - always raise/re-raise
    if hand in PREMIUM_HANDS or hand[:2] in {"AA", "KK", "QQ", "JJ"}:
        if to_call == 0:
            raise_amt = min(pot + min_raise, stack)
            return f"raise {raise_amt}"
        elif to_call < stack * 0.3:
            raise_amt = min(current_bet + pot, stack + current_bet)
            return f"raise {raise_amt}"
        else:
            return "call"  # Call big raises with premium hands

    # Strong hands - raise or call
    if hand in STRONG_HANDS:
        if to_call == 0:
            raise_amt = min(pot, stack)
            return f"raise {raise_amt}"
        elif to_call < stack * 0.15:
            return "call"
        elif in_position and to_call < stack * 0.25:
            return "call"
        else:
            return "fold"

    # Playable hands - play in position or cheaply
    if hand in PLAYABLE_HANDS:
        if to_call == 0:
            if in_position:
                raise_amt = min(pot, stack)
                return f"raise {raise_amt}"
            return "check"
        elif to_call < stack * 0.08:
            return "call"
        elif in_position and to_call < stack * 0.12:
            return "call"
        else:
            return "fold"

    # Weak hands - fold to bets, check if free
    if to_call == 0:
        return "check"
    return "fold"


def postflop_strategy(
    hole_cards, community_cards, to_call, pot, stack, min_raise, current_bet, round_name
) -> str:
    """Postflop betting strategy based on made hand strength."""
    hand_rank, _ = evaluate_hand_strength(hole_cards, community_cards)

    # Very strong hands (trips or better) - bet/raise aggressively
    if hand_rank >= 3:
        if to_call == 0:
            bet_size = int(pot * 0.75)
            return f"raise {bet_size}"
        elif to_call < pot:
            raise_amt = current_bet + pot
            return f"raise {raise_amt}"
        else:
            return "call"  # Call big bets with strong hands

    # Two pair - bet for value, call reasonable bets
    if hand_rank == 2:
        if to_call == 0:
            bet_size = int(pot * 0.6)
            return f"raise {bet_size}"
        elif to_call < pot * 0.6:
            return "call"
        else:
            return "fold"

    # One pair - cautious value betting
    if hand_rank == 1:
        RANKS = "23456789TJQKA"
        pair_rank = _get_pair_rank(hole_cards, community_cards)

        # Top pair or overpair
        if pair_rank is not None:
            board_ranks = [RANKS.index(c[0]) for c in community_cards]
            max_board = max(board_ranks) if board_ranks else -1

            if pair_rank >= max_board:  # Top pair or better
                if to_call == 0:
                    bet_size = int(pot * 0.5)
                    return f"raise {bet_size}"
                elif to_call < pot * 0.4:
                    return "call"
                else:
                    return "fold"
            else:  # Underpair
                if to_call == 0:
                    return "check"
                elif to_call < pot * 0.25:
                    return "call"
                else:
                    return "fold"

        # Pocket pair below board
        if to_call == 0:
            return "check"
        elif to_call < pot * 0.2:
            return "call"
        else:
            return "fold"

    # High card - check/fold mostly
    if to_call == 0:
        # Occasional bluff on the river
        if round_name == "river":
            import random

            if random.random() < 0.15:
                return f"raise {int(pot * 0.5)}"
        return "check"

    # Consider pot odds for draws
    if round_name in ("flop", "turn"):
        if has_draw(hole_cards, community_cards):
            pot_odds = to_call / (pot + to_call) if pot + to_call > 0 else 1
            if pot_odds < 0.25:  # Need about 4:1 odds
                return "call"

    return "fold"


def _get_pair_rank(hole_cards, community_cards):
    """Get the rank of the pair if we have one with our hole cards."""
    RANKS = "23456789TJQKA"
    hole_ranks = [RANKS.index(c[0]) for c in hole_cards]
    board_ranks = [RANKS.index(c[0]) for c in community_cards]

    # Pocket pair
    if hole_ranks[0] == hole_ranks[1]:
        return hole_ranks[0]

    # Pair with board
    for hr in hole_ranks:
        if hr in board_ranks:
            return hr

    return None


def has_draw(hole_cards, community_cards):
    """Check if we have a flush or straight draw."""
    all_cards = hole_cards + community_cards
    suits = [c[1] for c in all_cards]
    suit_counts = Counter(suits)

    # Flush draw (4 to a flush)
    if max(suit_counts.values()) >= 4:
        return True

    # Straight draw (open-ended or gutshot)
    RANKS = "23456789TJQKA"
    ranks = sorted(set(RANKS.index(c[0]) for c in all_cards))

    # Check for 4 consecutive or 4 with one gap
    for i in range(len(ranks) - 3):
        window = ranks[i : i + 4]
        if window[-1] - window[0] <= 4:
            return True

    return False


if __name__ == "__main__":
    # Test the bot
    class MockState:
        hole_cards = ["As", "Kh"]
        community_cards = []
        pot = 15
        current_bet = 10
        player_stack = 990
        opponent_stack = 990
        player_bet = 5
        opponent_bet = 10
        position = "button"
        round_name = "preflop"
        min_raise = 10
        is_first_action = False

    state = MockState()
    print(f"Hand: {get_hand_category(state.hole_cards)}")
    print(f"Action: {get_move(state)}")
