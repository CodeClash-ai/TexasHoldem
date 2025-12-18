#!/usr/bin/env python3
"""
Texas Hold'em Poker Engine for CodeClash

This engine runs heads-up (2-player) No-Limit Texas Hold'em poker games.
Supports both classic (52-card) and short-deck/six-plus (36-card) variants.

Each player bot receives game state and must return an action (fold, call, raise).
"""

import argparse
import importlib.util
import os
import random
import sys
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import combinations
from typing import Callable


# =============================================================================
# Hand Rankings
# =============================================================================

class HandRank(IntEnum):
    """Standard poker hand rankings."""
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9


class ShortDeckHandRank(IntEnum):
    """Short-deck hand rankings (flush beats full house)."""
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FULL_HOUSE = 5  # Full house is BELOW flush in short deck
    FLUSH = 6       # Flush is ABOVE full house in short deck
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9


# =============================================================================
# Hand Evaluator Classes
# =============================================================================

class HandEvaluator(ABC):
    """Abstract base class for poker hand evaluation."""

    RANKS: str  # Card ranks from lowest to highest
    SUITS: str = "cdhs"  # clubs, diamonds, hearts, spades

    @classmethod
    def create_deck(cls) -> list[str]:
        """Create a deck with the configured ranks."""
        return [r + s for r in cls.RANKS for s in cls.SUITS]

    @classmethod
    def rank_value(cls, card: str) -> int:
        """Get numeric value of card rank."""
        return cls.RANKS.index(card[0])

    @classmethod
    def suit_value(cls, card: str) -> str:
        """Get suit of card."""
        return card[1]

    @classmethod
    @abstractmethod
    def get_hand_rank_enum(cls) -> type[IntEnum]:
        """Return the appropriate HandRank enum for this evaluator."""
        pass

    @classmethod
    @abstractmethod
    def get_wheel_ranks(cls) -> list[int]:
        """Return the rank indices for the wheel (lowest straight)."""
        pass

    @classmethod
    def evaluate_hand(cls, cards: list[str]) -> tuple[IntEnum, list[int]]:
        """
        Evaluate a poker hand (5-7 cards) and return (HandRank, tiebreaker_values).
        Returns the best possible 5-card hand from the given cards.
        """
        if len(cards) < 5:
            raise ValueError("Need at least 5 cards to evaluate")

        HandRankEnum = cls.get_hand_rank_enum()
        best_rank = HandRankEnum.HIGH_CARD
        best_tiebreaker: list[int] = []

        for combo in combinations(cards, 5):
            rank, tiebreaker = cls._evaluate_five_cards(list(combo))
            if rank > best_rank or (rank == best_rank and tiebreaker > best_tiebreaker):
                best_rank = rank
                best_tiebreaker = tiebreaker

        return best_rank, best_tiebreaker

    @classmethod
    def _evaluate_five_cards(cls, cards: list[str]) -> tuple[IntEnum, list[int]]:
        """Evaluate exactly 5 cards."""
        HandRankEnum = cls.get_hand_rank_enum()

        ranks = sorted([cls.rank_value(c) for c in cards], reverse=True)
        suits = [cls.suit_value(c) for c in cards]
        rank_counts = Counter(ranks)
        is_flush = len(set(suits)) == 1

        # Check for straight
        sorted_ranks = sorted(set(ranks))
        is_straight = False
        straight_high = 0

        if len(sorted_ranks) == 5:
            if sorted_ranks[-1] - sorted_ranks[0] == 4:
                is_straight = True
                straight_high = sorted_ranks[-1]
            # Check for wheel (lowest straight)
            wheel_ranks = cls.get_wheel_ranks()
            if sorted_ranks == wheel_ranks:
                is_straight = True
                straight_high = wheel_ranks[-2]  # Second highest is the "high" for wheel

        counts = sorted(rank_counts.values(), reverse=True)

        # Determine hand rank
        if is_straight and is_flush:
            # Royal flush check: highest straight flush
            ace_index = len(cls.RANKS) - 1
            ten_index = cls.RANKS.index('T')
            if straight_high == ace_index and ten_index in ranks:
                return HandRankEnum.ROYAL_FLUSH, [straight_high]
            return HandRankEnum.STRAIGHT_FLUSH, [straight_high]

        if counts == [4, 1]:
            quad_rank = [r for r, c in rank_counts.items() if c == 4][0]
            kicker = [r for r, c in rank_counts.items() if c == 1][0]
            return HandRankEnum.FOUR_OF_A_KIND, [quad_rank, kicker]

        if counts == [3, 2]:
            trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
            pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
            return HandRankEnum.FULL_HOUSE, [trip_rank, pair_rank]

        if is_flush:
            return HandRankEnum.FLUSH, ranks

        if is_straight:
            return HandRankEnum.STRAIGHT, [straight_high]

        if counts == [3, 1, 1]:
            trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
            kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
            return HandRankEnum.THREE_OF_A_KIND, [trip_rank] + kickers

        if counts == [2, 2, 1]:
            pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
            kicker = [r for r, c in rank_counts.items() if c == 1][0]
            return HandRankEnum.TWO_PAIR, pairs + [kicker]

        if counts == [2, 1, 1, 1]:
            pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
            kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
            return HandRankEnum.PAIR, [pair_rank] + kickers

        return HandRankEnum.HIGH_CARD, ranks

    @classmethod
    def compare_hands(cls, hand1: list[str], hand2: list[str]) -> int:
        """
        Compare two hands. Returns:
        - 1 if hand1 wins
        - -1 if hand2 wins
        - 0 if tie
        """
        rank1, tie1 = cls.evaluate_hand(hand1)
        rank2, tie2 = cls.evaluate_hand(hand2)

        if rank1 > rank2:
            return 1
        if rank1 < rank2:
            return -1
        if tie1 > tie2:
            return 1
        if tie1 < tie2:
            return -1
        return 0


class ClassicHandEvaluator(HandEvaluator):
    """Hand evaluator for classic 52-card Texas Hold'em."""

    RANKS = "23456789TJQKA"  # 13 ranks × 4 suits = 52 cards

    @classmethod
    def get_hand_rank_enum(cls) -> type[IntEnum]:
        return HandRank

    @classmethod
    def get_wheel_ranks(cls) -> list[int]:
        # A-2-3-4-5 wheel: indices [0, 1, 2, 3, 12]
        return [0, 1, 2, 3, 12]


class ShortDeckHandEvaluator(HandEvaluator):
    """
    Hand evaluator for Short-deck (Six-plus) Hold'em.

    Key differences from classic:
    - 36-card deck (6 through Ace, removing 2-5)
    - Flush beats Full House (flushes are harder to make)
    - A-6-7-8-9 is the lowest straight (wheel)
    """

    RANKS = "6789TJQKA"  # 9 ranks × 4 suits = 36 cards

    @classmethod
    def get_hand_rank_enum(cls) -> type[IntEnum]:
        return ShortDeckHandRank

    @classmethod
    def get_wheel_ranks(cls) -> list[int]:
        # A-6-7-8-9 wheel in short deck: indices [0, 1, 2, 3, 8] (6=0, 7=1, 8=2, 9=3, A=8)
        return [0, 1, 2, 3, 8]


# =============================================================================
# Backward Compatibility - Module-level functions using classic evaluator
# =============================================================================

RANKS = ClassicHandEvaluator.RANKS
SUITS = ClassicHandEvaluator.SUITS


def create_deck() -> list[str]:
    """Create a standard 52-card deck."""
    return ClassicHandEvaluator.create_deck()


def rank_value(card: str) -> int:
    """Get numeric value of card rank (2=0, 3=1, ..., A=12)."""
    return ClassicHandEvaluator.rank_value(card)


def suit_value(card: str) -> str:
    """Get suit of card."""
    return ClassicHandEvaluator.suit_value(card)


def evaluate_hand(cards: list[str]) -> tuple[HandRank, list[int]]:
    """Evaluate a poker hand using classic rules."""
    return ClassicHandEvaluator.evaluate_hand(cards)


def compare_hands(hand1: list[str], hand2: list[str]) -> int:
    """Compare two hands using classic rules."""
    return ClassicHandEvaluator.compare_hands(hand1, hand2)


# For backward compatibility, keep the internal function
def _evaluate_five_cards(cards: list[str]) -> tuple[HandRank, list[int]]:
    """Evaluate exactly 5 cards using classic rules."""
    return ClassicHandEvaluator._evaluate_five_cards(cards)


# =============================================================================
# Game State and Player
# =============================================================================

@dataclass
class GameState:
    """State passed to bot's get_move function."""

    hole_cards: list[str]  # Player's 2 hole cards
    community_cards: list[str]  # Current community cards (0-5)
    pot: int  # Total pot size
    current_bet: int  # Current bet to call
    player_stack: int  # Your remaining chips
    opponent_stack: int  # Opponent's remaining chips
    player_bet: int  # Amount you've bet this round
    opponent_bet: int  # Amount opponent has bet this round
    position: str  # 'button' (acts first preflop, last postflop) or 'big_blind'
    round_name: str  # 'preflop', 'flop', 'turn', 'river'
    min_raise: int  # Minimum raise amount
    is_first_action: bool  # True if first to act this betting round
    variant: str = "classic"  # 'classic' or 'short_deck'


@dataclass
class Player:
    """A player in the game."""

    name: str
    get_move: Callable[[GameState], str]
    stack: int = 1000
    hole_cards: list[str] = field(default_factory=list)
    current_bet: int = 0
    folded: bool = False
    all_in: bool = False


# =============================================================================
# Game Classes
# =============================================================================

class TexasHoldemGame:
    """Manages a single Texas Hold'em hand (classic 52-card variant)."""

    SMALL_BLIND = 5
    BIG_BLIND = 10
    STARTING_STACK = 1000
    VARIANT = "classic"

    # Evaluator class to use (can be overridden by subclasses)
    evaluator: type[HandEvaluator] = ClassicHandEvaluator

    def __init__(self, player1_move: Callable, player2_move: Callable, verbose: bool = False):
        self.players = [
            Player("player1", player1_move),
            Player("player2", player2_move),
        ]
        self.deck: list[str] = []
        self.community_cards: list[str] = []
        self.pot = 0
        self.current_bet = 0
        self.min_raise = self.BIG_BLIND
        self.verbose = verbose
        self.button = 0  # Index of button player

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def reset_hand(self):
        """Reset for a new hand."""
        self.deck = self.evaluator.create_deck()
        random.shuffle(self.deck)
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.min_raise = self.BIG_BLIND

        for p in self.players:
            p.hole_cards = []
            p.current_bet = 0
            p.folded = False
            p.all_in = False

    def deal_hole_cards(self):
        """Deal 2 hole cards to each player."""
        for p in self.players:
            p.hole_cards = [self.deck.pop(), self.deck.pop()]

    def deal_community(self, count: int):
        """Deal community cards."""
        # Burn one card
        self.deck.pop()
        for _ in range(count):
            self.community_cards.append(self.deck.pop())

    def post_blinds(self):
        """Post small and big blinds."""
        sb_player = self.players[self.button]
        bb_player = self.players[1 - self.button]

        sb_amount = min(self.SMALL_BLIND, sb_player.stack)
        bb_amount = min(self.BIG_BLIND, bb_player.stack)

        sb_player.stack -= sb_amount
        sb_player.current_bet = sb_amount

        bb_player.stack -= bb_amount
        bb_player.current_bet = bb_amount

        self.pot = sb_amount + bb_amount
        self.current_bet = bb_amount

        self.log(f"{sb_player.name} posts small blind {sb_amount}")
        self.log(f"{bb_player.name} posts big blind {bb_amount}")

    def get_action(self, player: Player, opponent: Player, round_name: str, is_first: bool) -> str:
        """Get action from player's bot."""
        position = "button" if self.players.index(player) == self.button else "big_blind"

        state = GameState(
            hole_cards=player.hole_cards.copy(),
            community_cards=self.community_cards.copy(),
            pot=self.pot,
            current_bet=self.current_bet,
            player_stack=player.stack,
            opponent_stack=opponent.stack,
            player_bet=player.current_bet,
            opponent_bet=opponent.current_bet,
            position=position,
            round_name=round_name,
            min_raise=self.min_raise,
            is_first_action=is_first,
            variant=self.VARIANT,
        )

        try:
            action = player.get_move(state)
            return self.parse_action(action, player)
        except Exception as e:
            self.log(f"Error from {player.name}: {e}")
            return "fold"

    def parse_action(self, action: str, player: Player) -> str:
        """Parse and validate player action."""
        action = action.lower().strip()

        if action == "fold":
            return "fold"

        if action in ("call", "check"):
            to_call = self.current_bet - player.current_bet
            if to_call == 0:
                return "check"
            return "call"

        if action.startswith("raise"):
            parts = action.split()
            if len(parts) == 2:
                try:
                    amount = int(parts[1])
                    return f"raise {amount}"
                except ValueError:
                    pass
            # Default raise to min raise
            return f"raise {self.current_bet + self.min_raise}"

        if action == "all_in" or action == "allin":
            return f"raise {player.stack + player.current_bet}"

        # Default to check/call
        to_call = self.current_bet - player.current_bet
        return "check" if to_call == 0 else "call"

    def execute_action(self, player: Player, action: str) -> bool:
        """Execute player action. Returns True if betting round continues."""
        if action == "fold":
            player.folded = True
            self.log(f"{player.name} folds")
            return False

        if action == "check":
            self.log(f"{player.name} checks")
            return True

        if action == "call":
            to_call = self.current_bet - player.current_bet
            call_amount = min(to_call, player.stack)
            player.stack -= call_amount
            player.current_bet += call_amount
            self.pot += call_amount
            if player.stack == 0:
                player.all_in = True
            self.log(f"{player.name} calls {call_amount}")
            return True

        if action.startswith("raise"):
            parts = action.split()
            target_bet = int(parts[1])

            # Ensure raise is valid
            to_call = self.current_bet - player.current_bet
            min_target = self.current_bet + self.min_raise

            if target_bet < min_target and target_bet < player.stack + player.current_bet:
                target_bet = min_target

            raise_amount = target_bet - player.current_bet
            raise_amount = min(raise_amount, player.stack)

            if raise_amount > to_call:
                self.min_raise = raise_amount - to_call

            player.stack -= raise_amount
            self.pot += raise_amount
            actual_target = player.current_bet + raise_amount
            player.current_bet = actual_target
            self.current_bet = actual_target

            if player.stack == 0:
                player.all_in = True
                self.log(f"{player.name} goes all-in for {raise_amount}")
            else:
                self.log(f"{player.name} raises to {actual_target}")

            return True

        return True

    def betting_round(self, round_name: str, first_to_act: int):
        """Run a betting round. Returns True if hand continues."""
        # Reset per-round bets
        for p in self.players:
            p.current_bet = 0
        self.current_bet = 0
        self.min_raise = self.BIG_BLIND

        # If preflop, blinds already posted
        if round_name == "preflop":
            sb = self.players[self.button]
            bb = self.players[1 - self.button]
            sb.current_bet = self.SMALL_BLIND
            bb.current_bet = self.BIG_BLIND
            self.current_bet = self.BIG_BLIND

        active = [p for p in self.players if not p.folded and not p.all_in]
        if len(active) <= 1:
            return len([p for p in self.players if not p.folded]) > 1

        # Determine action order
        current = first_to_act
        last_raiser = -1
        actions_taken = 0
        first_action = [True, True]

        while True:
            player = self.players[current]
            opponent = self.players[1 - current]

            if player.folded or player.all_in:
                current = 1 - current
                if current == last_raiser or (last_raiser == -1 and actions_taken >= 2):
                    break
                continue

            action = self.get_action(player, opponent, round_name, first_action[current])
            first_action[current] = False
            continues = self.execute_action(player, action)
            actions_taken += 1

            if not continues:
                return False  # Player folded

            if action.startswith("raise"):
                last_raiser = current

            current = 1 - current

            # Check if betting is complete
            active_players = [p for p in self.players if not p.folded and not p.all_in]
            if len(active_players) == 0:
                break
            if len(active_players) == 1 and self.players[current].all_in:
                break

            # Both have acted and bets are equal
            bets_equal = self.players[0].current_bet == self.players[1].current_bet
            both_acted = actions_taken >= 2

            if bets_equal and both_acted and current == last_raiser:
                break
            if bets_equal and both_acted and last_raiser == -1:
                break

        return True

    def showdown(self) -> int:
        """Determine winner at showdown. Returns player index or -1 for tie."""
        p1, p2 = self.players

        if p1.folded:
            return 1
        if p2.folded:
            return 0

        hand1 = p1.hole_cards + self.community_cards
        hand2 = p2.hole_cards + self.community_cards

        result = self.evaluator.compare_hands(hand1, hand2)

        self.log("Showdown:")
        self.log(f"  {p1.name}: {p1.hole_cards} -> {self.evaluator.evaluate_hand(hand1)}")
        self.log(f"  {p2.name}: {p2.hole_cards} -> {self.evaluator.evaluate_hand(hand2)}")

        if result > 0:
            return 0
        if result < 0:
            return 1
        return -1  # Tie

    def play_hand(self) -> int:
        """Play a single hand. Returns winner index or -1 for tie."""
        self.reset_hand()
        self.deal_hole_cards()
        self.post_blinds()

        self.log(f"\n=== New Hand ({self.VARIANT}) ===")
        self.log(f"Button: {self.players[self.button].name}")
        self.log(f"{self.players[0].name}: {self.players[0].hole_cards}")
        self.log(f"{self.players[1].name}: {self.players[1].hole_cards}")

        # Preflop - button acts first
        if not self.betting_round("preflop", self.button):
            winner = 0 if self.players[1].folded else 1
            self.players[winner].stack += self.pot
            self.log(f"{self.players[winner].name} wins {self.pot}")
            return winner

        # Flop - BB acts first (non-button)
        self.deal_community(3)
        self.log(f"Flop: {self.community_cards}")
        if not self.betting_round("flop", 1 - self.button):
            winner = 0 if self.players[1].folded else 1
            self.players[winner].stack += self.pot
            self.log(f"{self.players[winner].name} wins {self.pot}")
            return winner

        # Turn
        self.deal_community(1)
        self.log(f"Turn: {self.community_cards}")
        if not self.betting_round("turn", 1 - self.button):
            winner = 0 if self.players[1].folded else 1
            self.players[winner].stack += self.pot
            self.log(f"{self.players[winner].name} wins {self.pot}")
            return winner

        # River
        self.deal_community(1)
        self.log(f"River: {self.community_cards}")
        if not self.betting_round("river", 1 - self.button):
            winner = 0 if self.players[1].folded else 1
            self.players[winner].stack += self.pot
            self.log(f"{self.players[winner].name} wins {self.pot}")
            return winner

        # Showdown
        winner = self.showdown()
        if winner >= 0:
            self.players[winner].stack += self.pot
            self.log(f"{self.players[winner].name} wins {self.pot}")
        else:
            # Split pot
            split = self.pot // 2
            self.players[0].stack += split
            self.players[1].stack += self.pot - split
            self.log(f"Pot split: {split} each")

        return winner


class ShortDeckHoldemGame(TexasHoldemGame):
    """
    Short-deck (Six-plus) Hold'em variant.

    Key differences:
    - 36-card deck (6 through Ace)
    - Flush beats Full House
    - A-6-7-8-9 is the lowest straight
    """

    VARIANT = "short_deck"
    evaluator: type[HandEvaluator] = ShortDeckHandEvaluator


# =============================================================================
# Bot Loading and Main
# =============================================================================

def load_bot(bot_path: str) -> Callable:
    """Load a bot's get_move function from a Python file."""
    spec = importlib.util.spec_from_file_location("bot_module", bot_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load bot from {bot_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["bot_module"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "get_move"):
        raise AttributeError(f"Bot at {bot_path} must define a 'get_move' function")

    return module.get_move


def main():
    parser = argparse.ArgumentParser(description="Texas Hold'em Poker Engine")
    parser.add_argument("players", nargs=2, help="Paths to player bot files")
    parser.add_argument("-r", "--rounds", type=int, default=100, help="Number of hands to play")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed game log")
    parser.add_argument(
        "--variant",
        choices=["classic", "short_deck"],
        default="classic",
        help="Game variant: classic (52-card) or short_deck (36-card, 6-A)",
    )
    args = parser.parse_args()

    # Load bot modules
    bot1 = load_bot(args.players[0])
    bot2 = load_bot(args.players[1])

    # Select game class based on variant
    if args.variant == "short_deck":
        game = ShortDeckHoldemGame(bot1, bot2, verbose=args.verbose)
    else:
        game = TexasHoldemGame(bot1, bot2, verbose=args.verbose)

    scores = {"player1": 0, "player2": 0, "draw": 0}

    for hand_num in range(args.rounds):
        winner = game.play_hand()

        if winner == 0:
            scores["player1"] += 1
        elif winner == 1:
            scores["player2"] += 1
        else:
            scores["draw"] += 1

        # Alternate button
        game.button = 1 - game.button

        # Reset stacks for next hand
        game.players[0].stack = game.STARTING_STACK
        game.players[1].stack = game.STARTING_STACK

    # Output in required format
    print()
    print("FINAL_RESULTS")
    for i, path in enumerate(args.players):
        name = os.path.basename(os.path.dirname(path))
        print(f"Bot_{i+1}_main: {scores[f'player{i+1}']} rounds won ({name})")
    print(f"Draws: {scores['draw']}")


if __name__ == "__main__":
    main()
