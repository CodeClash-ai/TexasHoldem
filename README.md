# Texas Hold'em - CodeClash

A heads-up (2-player) No-Limit Texas Hold'em poker engine for CodeClash AI competitions.

## Game Overview

Texas Hold'em is the most popular poker variant worldwide. In this heads-up format:

1. Each player receives 2 private "hole" cards
2. 5 community cards are dealt face-up in stages (Flop: 3, Turn: 1, River: 1)
3. Players bet/raise/fold based on hand strength
4. Best 5-card hand from 7 available cards wins

### Hand Rankings (Highest to Lowest)

| Rank | Hand | Example |
|------|------|---------|
| 10 | Royal Flush | A-K-Q-J-10 (same suit) |
| 9 | Straight Flush | 9-8-7-6-5 (same suit) |
| 8 | Four of a Kind | K-K-K-K-x |
| 7 | Full House | Q-Q-Q-7-7 |
| 6 | Flush | Any 5 cards same suit |
| 5 | Straight | 8-7-6-5-4 (any suits) |
| 4 | Three of a Kind | J-J-J-x-x |
| 3 | Two Pair | A-A-8-8-x |
| 2 | One Pair | K-K-x-x-x |
| 1 | High Card | A-K-Q-J-9 |

### Betting Structure

- **Blinds**: Small blind = 5, Big blind = 10
- **Starting Stack**: 1000 chips per hand
- **No-Limit**: Players can bet any amount up to their stack

## Repository Structure

```
TexasHoldem/
├── engine.py     # Game engine (run this)
├── main.py       # Starter bot (your submission file)
└── README.md     # This file
```

## Bot Interface

Your bot must implement a single function:

```python
def get_move(state) -> str:
    """
    Decide your action based on the current game state.

    Args:
        state: GameState object with the following attributes:
            - hole_cards: list[str]      # Your 2 hole cards, e.g., ['As', 'Kh']
            - community_cards: list[str] # Current community cards (0-5)
            - pot: int                   # Total pot size
            - current_bet: int           # Current bet to call
            - player_stack: int          # Your remaining chips
            - opponent_stack: int        # Opponent's remaining chips
            - player_bet: int            # Amount you've bet this round
            - opponent_bet: int          # Amount opponent has bet this round
            - position: str              # 'button' (dealer) or 'big_blind'
            - round_name: str            # 'preflop', 'flop', 'turn', 'river'
            - min_raise: int             # Minimum raise amount
            - is_first_action: bool      # True if first to act this betting round

    Returns:
        str: One of the following actions:
            - 'fold'          # Give up the hand
            - 'check'         # Pass (only when no bet to call)
            - 'call'          # Match the current bet
            - 'raise <amount>'# Raise to specified total amount
            - 'all_in'        # Bet all remaining chips
    """
    pass
```

### Card Notation

Cards are represented as 2-character strings:
- **Ranks**: `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `T` (10), `J`, `Q`, `K`, `A`
- **Suits**: `c` (clubs), `d` (diamonds), `h` (hearts), `s` (spades)

Examples: `'As'` = Ace of spades, `'Th'` = Ten of hearts, `'2c'` = Two of clubs

### Example Game State

```python
state.hole_cards = ['Ah', 'Kd']        # You have Ace-King
state.community_cards = ['Qh', 'Jh', 'Ts']  # Flop showing Q-J-T
state.pot = 120                        # 120 chips in the pot
state.current_bet = 40                 # 40 to call
state.player_stack = 960               # You have 960 chips
state.player_bet = 0                   # You haven't bet this round yet
state.position = 'button'              # You're in position
state.round_name = 'flop'              # On the flop
state.min_raise = 40                   # Minimum raise is 40 more
```

## Running Games Locally

```bash
# Run a quick test (5 hands)
python engine.py main.py main.py -r 5

# Run with verbose output
python engine.py main.py main.py -r 10 -v

# Run a full match (100 hands)
python engine.py path/to/bot1.py path/to/bot2.py -r 100
```

### Command Line Arguments

```
usage: engine.py [-h] [-r ROUNDS] [-v] players players

positional arguments:
  players               Paths to player bot files

optional arguments:
  -r, --rounds ROUNDS   Number of hands to play (default: 100)
  -v, --verbose         Print detailed game log
```

## Strategy Tips

### Preflop Hand Selection

**Premium hands** (always play aggressively):
- Pocket Aces (AA), Kings (KK), Queens (QQ), Jacks (JJ)
- Ace-King suited or offsuit

**Strong hands** (raise or call raises):
- TT, 99, AQ, AJ, KQ suited

**Playable hands** (play in position):
- Medium pairs (88-66), suited connectors (JTs, T9s)
- Suited Aces (A5s-A2s)

### Position Matters

- **Button (Dealer)**: Acts last postflop - play more hands
- **Big Blind**: Acts first postflop - play tighter

### Pot Odds

Calculate if calling is profitable:
```
Pot odds = amount_to_call / (pot + amount_to_call)
```

If your estimated chance of winning > pot odds, call is profitable.

### Key Concepts

1. **Value Betting**: Bet strong hands to get called by worse hands
2. **Bluffing**: Bet weak hands to make better hands fold
3. **Position**: Being last to act is a significant advantage
4. **Hand Reading**: Use opponent's actions to narrow their range
5. **Stack Management**: Adjust bet sizes based on stack depth

## Output Format

The engine outputs results in CodeClash format:

```
FINAL_RESULTS
Bot_1_main: 55 rounds won (player1)
Bot_2_main: 42 rounds won (player2)
Draws: 3
```

## License

MIT License - See CodeClash repository for details.
