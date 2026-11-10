# River Toy-Game Solver

A small river-only GTO solver (Discounted CFR) for studying poker theory with toy games.
Heads-up or multiway, any ranges, any bet sizes.

## In the browser
Open **https://mrjayis.github.io/river-toy-solver/**, or open `index.html` locally. There's nothing to install,
and the solver runs entirely in your browser.

## Terminal version setup
    python3 -m venv .venv && .venv/bin/pip install numpy

## Run
Prompted mode (asks for everything):

    .venv/bin/python river_solver.py

Command-line mode (the classic AAA22 polar-vs-bluffcatcher game):

    .venv/bin/python river_solver.py --board AsAdAh2s2d --pot 10 --stack 10 \
        --range "KK,33" --range "88" --bets 100

Three-way: add another `--range`. Players act in the order the ranges are given (first = OOP).

## Options
| flag | meaning | default |
|---|---|---|
| `--bets 33,75,150` | bet sizes, % of pot | 50,100 |
| `--raises 100` | raise sizes, % of pot after calling | 100 |
| `--max-raises 1` | raises allowed after the first bet (0 = no raising) | 1 |
| `--no-allin` | don't add an all-in option automatically | all-in on |
| `--iters`, `--target` | max iterations / stop at this exploitability (% pot) | 2000 / 0.1 |
| `--names Hero,Villain` | player labels | OOP/IP or P1..Pn |
| `--no-browse` | print the root and exit | |

## Range syntax
`KK`, `AKs`, `AKo`, `AK`, `TT+`, `A9s+`, `22-55`, `A2s-A5s`, specific combos `AsKd`,
`any`, and weights like `KK:0.5`. Comma separated. Combos blocked by the board are removed,
and card removal between players is handled exactly.

## Reading the output
After solving you get a tree browser: type an action number to go down a line, `b` back,
`r` root, `c` to toggle hand classes vs individual combos, `q` to quit.

For the player to act, each row shows the strategy, the hand's EV, and the EV of each action.
At equilibrium every action a hand mixes between has the same EV (indifference),
which is the easiest way to see the theory working.

EVs are in chips measured from the start of the river, so they include each player's share
of the starting pot (heads-up, OOP EV + IP EV = pot), and folding after money has gone in
shows as a loss of that money.

## Limitations
* One effective stack for everyone (no side pots).
* Same bet/raise sizes for every player.
* Multiway: CFR isn't guaranteed to reach a Nash equilibrium with 3+ players, but the
  "could gain by deviating" number tells you how close each player is to unexploitable.
* Joint range size is capped (~30M combo tuples), which is plenty for toy games and
  normal heads-up river ranges.
