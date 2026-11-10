#!/usr/bin/env python3
"""
River toy-game solver (heads-up or multiway).

Solves a single river betting round to an approximate Nash equilibrium using
Discounted CFR (DCFR), the same family of algorithms that commercial solvers use.

Example (the classic AAA22 toy game):
    python river_solver.py --board AsAdAh2s2d --pot 10 --stack 10 \
        --range "KK,33" --range "88" --bets 100

Run with no arguments to be prompted for every input.

Conventions
-----------
* Players are listed in order of action: the first --range is first to act (OOP).
* --pot is the pot at the start of the river; --stack is the effective stack
  behind for every player (no side pots).
* Bet sizes are % of the current pot. Raise sizes are % of the pot *after*
  calling (the same convention as PioSolver).
* EVs are in chips and include each player's share of the starting pot, so in
  heads-up OOP EV + IP EV = starting pot.
"""

import argparse
import itertools
import sys
import time
from collections import Counter

import numpy as np

# --------------------------------------------------------------------------
# Cards
# --------------------------------------------------------------------------
RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card_from_str(s):
    s = s.strip()
    if len(s) != 2 or s[0].upper() not in RANKS or s[1].lower() not in SUITS:
        raise ValueError(f"bad card '{s}'")
    return RANKS.index(s[0].upper()) * 4 + SUITS.index(s[1].lower())


def card_str(c):
    return RANKS[c // 4] + SUITS[c % 4]


def parse_cards(s):
    s = s.replace(" ", "").replace(",", "")
    if len(s) % 2:
        raise ValueError(f"bad card string '{s}'")
    return [card_from_str(s[i:i + 2]) for i in range(0, len(s), 2)]


# --------------------------------------------------------------------------
# Hand evaluation (best 5 of 7, returns a comparable tuple)
# --------------------------------------------------------------------------
def eval5(cards):
    ranks = sorted((c // 4 + 2 for c in cards), reverse=True)
    flush = len({c % 4 for c in cards}) == 1
    groups = sorted(Counter(ranks).items(), key=lambda x: (x[1], x[0]), reverse=True)
    uniq = sorted(set(ranks), reverse=True)
    straight = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight = uniq[0]
        elif uniq == [14, 5, 4, 3, 2]:
            straight = 5
    if straight and flush:
        return (8, straight)
    if groups[0][1] == 4:
        return (7, groups[0][0], groups[1][0])
    if groups[0][1] == 3 and groups[1][1] == 2:
        return (6, groups[0][0], groups[1][0])
    if flush:
        return (5, *ranks)
    if straight:
        return (4, straight)
    if groups[0][1] == 3:
        return (3, *[g[0] for g in groups])
    if groups[0][1] == 2 and groups[1][1] == 2:
        return (2, *[g[0] for g in groups])
    if groups[0][1] == 2:
        return (1, *[g[0] for g in groups])
    return (0, *ranks)


def eval7(cards):
    return max(eval5(c) for c in itertools.combinations(cards, 5))


# --------------------------------------------------------------------------
# Range parsing
# --------------------------------------------------------------------------
def _pair(r):
    cs = [r * 4 + s for s in range(4)]
    return [tuple(sorted(x)) for x in itertools.combinations(cs, 2)]


def _suited(r1, r2):
    return [tuple(sorted((r1 * 4 + s, r2 * 4 + s))) for s in range(4)]


def _offsuit(r1, r2):
    return [tuple(sorted((r1 * 4 + s1, r2 * 4 + s2)))
            for s1 in range(4) for s2 in range(4) if s1 != s2]


def _class_combos(hi, lo, kind):
    if hi == lo:
        return _pair(hi)
    if kind == "s":
        return _suited(hi, lo)
    if kind == "o":
        return _offsuit(hi, lo)
    return _suited(hi, lo) + _offsuit(hi, lo)


def _parse_class(t):
    """'AKs' -> (12, 11, 's'); 'KK' -> (11, 11, ''); 'AK' -> (12, 11, '')."""
    t = t.strip()
    if len(t) not in (2, 3) or t[0].upper() not in RANKS or t[1].upper() not in RANKS:
        raise ValueError(f"bad hand '{t}'")
    a, b = RANKS.index(t[0].upper()), RANKS.index(t[1].upper())
    kind = t[2].lower() if len(t) == 3 else ""
    if kind not in ("", "s", "o"):
        raise ValueError(f"bad hand '{t}'")
    if a == b and kind:
        raise ValueError(f"pairs can't be suited/offsuit: '{t}'")
    return max(a, b), min(a, b), kind


def expand_token(tok):
    """Expand one range token into a list of 2-card combos (sorted tuples)."""
    tok = tok.strip()
    low = tok.lower()
    if low in ("any", "random", "*", "100%"):
        return [tuple(x) for x in itertools.combinations(range(52), 2)]
    # specific combo, e.g. AsKd
    if len(tok) == 4 and tok[1].lower() in SUITS and tok[3].lower() in SUITS:
        c1, c2 = card_from_str(tok[:2]), card_from_str(tok[2:])
        if c1 == c2:
            raise ValueError(f"bad combo '{tok}'")
        return [tuple(sorted((c1, c2)))]
    # dash range: 22-55, A2s-A5s
    if "-" in tok:
        a, b = tok.split("-", 1)
        h1, l1, k1 = _parse_class(a)
        h2, l2, k2 = _parse_class(b)
        if k1 != k2:
            raise ValueError(f"bad range '{tok}'")
        out = []
        if h1 == l1 and h2 == l2:
            for r in range(min(h1, h2), max(h1, h2) + 1):
                out += _pair(r)
        elif h1 == h2:
            for r in range(min(l1, l2), max(l1, l2) + 1):
                out += _class_combos(h1, r, k1)
        else:
            raise ValueError(f"bad range '{tok}'")
        return out
    # plus: TT+, A9s+
    if tok.endswith("+"):
        h, l, k = _parse_class(tok[:-1])
        out = []
        if h == l:
            for r in range(h, 13):
                out += _pair(r)
        else:
            for r in range(l, h):
                out += _class_combos(h, r, k)
        return out
    h, l, k = _parse_class(tok)
    return _class_combos(h, l, k)


def parse_range(text, board):
    """Returns dict combo -> weight, with board-blocked combos removed.
    Tokens are comma separated; optional weight suffix 'KK:0.5'."""
    combos = {}
    board_set = set(board)
    for raw in text.replace(" ", "").split(","):
        if not raw:
            continue
        weight = 1.0
        if ":" in raw:
            raw, w = raw.split(":", 1)
            weight = float(w)
        for c in expand_token(raw):
            if c[0] in board_set or c[1] in board_set:
                continue
            combos[c] = weight
    combos = {c: w for c, w in combos.items() if w > 0}
    if not combos:
        raise ValueError(f"range '{text}' has no combos left on this board")
    return combos


def hand_class(combo):
    a, b = combo
    ra, rb = a // 4, b // 4
    hi, lo = max(ra, rb), min(ra, rb)
    if hi == lo:
        return RANKS[hi] * 2
    return RANKS[hi] + RANKS[lo] + ("s" if a % 4 == b % 4 else "o")


def combo_str(combo):
    a, b = sorted(combo, reverse=True)
    return card_str(a) + card_str(b)


# --------------------------------------------------------------------------
# Game tree
# --------------------------------------------------------------------------
def fmt(x):
    return f"{x:.2f}".rstrip("0").rstrip(".")


class Node:
    def __init__(self):
        self.terminal = False
        self.player = None
        self.labels = []
        self.children = []
        # terminal info
        self.kind = None          # 'fold' or 'showdown'
        self.active = None
        # common
        self.contribs = None
        self.pot = 0.0
        # CFR tables
        self.regret = None
        self.strat_sum = None


class TreeConfig:
    def __init__(self, n, pot, stack, bets, raises, max_raises, allin):
        self.n, self.pot, self.stack = n, pot, stack
        self.bets, self.raises = bets, raises
        self.max_raises, self.allin = max_raises, allin


def build_tree(cfg):
    n, S, eps = cfg.n, cfg.stack, 1e-9

    def rec(contribs, active, acted, nxt, nraises):
        node = Node()
        node.contribs = contribs
        node.pot = cfg.pot + sum(contribs)
        node.active = active
        alive = [i for i in range(n) if active[i]]
        if len(alive) == 1:
            node.terminal, node.kind = True, "fold"
            return node
        bet = max(contribs)
        pending = [i for i in range(n)
                   if active[i] and contribs[i] < S - eps
                   and (not acted[i] or contribs[i] < bet - eps)]
        if not pending:
            node.terminal, node.kind = True, "showdown"
            return node
        p = next((nxt + k) % n for k in range(n) if (nxt + k) % n in pending)
        node.player = p
        to_call = bet - contribs[p]
        nx = (p + 1) % n

        def aggressive_levels(base_levels):
            levels = []
            for lv in base_levels:
                lv = min(lv, S)
                if lv > bet + eps and all(abs(lv - x) > 1e-6 for x in levels):
                    levels.append(lv)
            if cfg.allin and S > bet + eps and all(abs(S - x) > 1e-6 for x in levels):
                levels.append(S)
            return sorted(levels)

        def go(new_c, new_active, new_acted, new_raises):
            return rec(tuple(new_c), tuple(new_active), tuple(new_acted), nx, new_raises)

        if to_call <= eps:
            acted2 = list(acted); acted2[p] = True
            node.labels.append("Check")
            node.children.append(go(contribs, active, acted2, nraises))
            if bet <= eps:   # opening bet
                levels = aggressive_levels([b / 100 * node.pot for b in cfg.bets])
                for lv in levels:
                    c2 = list(contribs); c2[p] = lv
                    a2 = [False] * n; a2[p] = True
                    lab = (f"All-in {fmt(lv)}" if lv >= S - eps else "Bet") + \
                          ("" if lv >= S - eps else f" {fmt(lv)}")
                    lab += f" ({fmt(100 * lv / node.pot)}%)"
                    node.labels.append(lab)
                    node.children.append(go(c2, active, a2, nraises))
        else:
            act2 = list(active); act2[p] = False
            acted2 = list(acted); acted2[p] = True
            node.labels.append("Fold")
            node.children.append(go(contribs, act2, acted2, nraises))
            c2 = list(contribs); c2[p] = min(bet, S)
            node.labels.append(f"Call {fmt(min(bet, S) - contribs[p])}")
            node.children.append(go(c2, active, acted2, nraises))
            if nraises < cfg.max_raises and bet < S - eps:
                pot_after_call = node.pot + to_call
                levels = aggressive_levels([bet + r / 100 * pot_after_call for r in cfg.raises])
                for lv in levels:
                    c2 = list(contribs); c2[p] = lv
                    a2 = [False] * n; a2[p] = True
                    lab = f"All-in {fmt(lv)}" if lv >= S - eps else f"Raise to {fmt(lv)}"
                    node.labels.append(lab)
                    node.children.append(go(c2, active, a2, nraises + 1))
        return node

    return rec(tuple([0.0] * n), tuple([True] * n), tuple([False] * n), 0, 0)


def iter_nodes(node):
    yield node
    for c in node.children:
        yield from iter_nodes(c)


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------
class RiverSolver:
    def __init__(self, board, ranges, cfg, names=None):
        self.board = board
        self.cfg = cfg
        self.n = n = len(ranges)
        self.names = names or (["OOP", "IP"] if n == 2 else [f"P{i + 1}" for i in range(n)])

        # combos, weights, hand strengths
        self.combos, self.weights = [], []
        for r in ranges:
            cs = sorted(r.keys())
            self.combos.append(cs)
            self.weights.append(np.array([r[c] for c in cs], dtype=float))
        strengths = [[eval7(list(c) + board) for c in cs] for cs in self.combos]
        uniq = sorted({s for row in strengths for s in row})
        idx = {s: i for i, s in enumerate(uniq)}
        self.strength = [np.array([idx[s] for s in row]) for row in strengths]
        self.hand_desc = [[_describe(s) for s in row] for row in strengths]

        size = 1
        for cs in self.combos:
            size *= len(cs)
        if size > 30_000_000:
            raise ValueError(f"joint range too large ({size:,} combo tuples); "
                             "narrow the ranges for this toy solver")

        # prior tensor P[c0, c1, ...]: product of weights, zero where cards collide
        masks = [np.array([(1 << a) | (1 << b) for a, b in cs], dtype=np.int64)
                 for cs in self.combos]
        P = np.ones([len(cs) for cs in self.combos])
        for i in range(n):
            P = P * self._axis(self.weights[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                ok = (masks[i][:, None] & masks[j][None, :]) == 0
                shape = [1] * n
                shape[i], shape[j] = len(masks[i]), len(masks[j])
                P = P * ok.reshape(shape)
        if P.sum() <= 0:
            raise ValueError("ranges have no compatible card combinations")
        self.P = P
        self.Z = P.sum()
        self._share_cache = {}

        self.root = build_tree(cfg)
        for node in iter_nodes(self.root):
            if not node.terminal:
                m = len(self.combos[node.player])
                node.regret = np.zeros((len(node.children), m))
                node.strat_sum = np.zeros((len(node.children), m))
        self.iterations = 0

    def _axis(self, v, i):
        shape = [1] * self.n
        shape[i] = len(v)
        return v.reshape(shape)

    def _contract(self, T, reach, i):
        """Sum T over all axes except i, weighting axis j by reach[j]."""
        out = T
        for j in reversed(range(self.n)):
            if j != i:
                out = np.tensordot(out, reach[j], axes=([j], [0]))
        return out

    def _share_tensors(self, active):
        """For a showdown among `active` players: P * (fraction of pot player i wins)."""
        if active not in self._share_cache:
            alive = [i for i in range(self.n) if active[i]]
            strs = {i: self._axis(self.strength[i], i) for i in alive}
            best = None
            for i in alive:
                best = strs[i] if best is None else np.maximum(best, strs[i])
            winners = sum((strs[i] == best).astype(float) for i in alive)
            self._share_cache[active] = [
                self.P * ((strs[i] == best) / winners) if active[i] else None
                for i in range(self.n)]
        return self._share_cache[active]

    def terminal_values(self, node, reach, only=None):
        """Counterfactual values per combo for each player at a terminal node."""
        players = range(self.n) if only is None else [only]
        out = [None] * self.n
        if node.kind == "fold":
            winner = node.active.index(True)
            for i in players:
                u = (node.pot if i == winner else 0.0) - node.contribs[i]
                out[i] = u * self._contract(self.P, reach, i)
        else:
            shares = self._share_tensors(node.active)
            for i in players:
                mass = self._contract(self.P, reach, i)
                if shares[i] is None:
                    out[i] = -node.contribs[i] * mass
                else:
                    out[i] = node.pot * self._contract(shares[i], reach, i) - node.contribs[i] * mass
        return out

    @staticmethod
    def _regret_match(R):
        pos = np.maximum(R, 0)
        s = pos.sum(axis=0)
        A = R.shape[0]
        return np.where(s > 0, pos / np.where(s > 0, s, 1), 1.0 / A)

    @staticmethod
    def avg_strategy(node):
        s = node.strat_sum.sum(axis=0)
        A = node.strat_sum.shape[0]
        return np.where(s > 0, node.strat_sum / np.where(s > 0, s, 1), 1.0 / A)

    # ---- DCFR iteration (alternating updates: only player i learns this pass) ----
    def _cfr(self, node, reach, t, i):
        """Returns player i's counterfactual values per combo."""
        if node.terminal:
            return self.terminal_values(node, reach, only=i)[i]
        p = node.player
        sigma = self._regret_match(node.regret)
        if p != i:
            total = np.zeros(len(self.combos[i]))
            for a, child in enumerate(node.children):
                r2 = list(reach)
                r2[p] = reach[p] * sigma[a]
                total += self._cfr(child, r2, t, i)
            return total
        action_vals = np.array([self._cfr(c, reach, t, i) for c in node.children])
        value = (sigma * action_vals).sum(axis=0)

        # DCFR discounting (alpha=1.5, beta=0, gamma=2)
        R = node.regret
        pos_w = t ** 1.5 / (t ** 1.5 + 1)
        R *= np.where(R > 0, pos_w, 0.5)
        R += action_vals - value
        node.strat_sum *= (t / (t + 1)) ** 2
        node.strat_sum += reach[p] * sigma
        return value

    def _ev(self, node, reach):
        """Values for all players when everyone plays the average strategy."""
        if node.terminal:
            return self.terminal_values(node, reach)
        p = node.player
        avg = self.avg_strategy(node)
        totals = [np.zeros(len(self.combos[j])) for j in range(self.n)]
        for a, child in enumerate(node.children):
            r2 = list(reach)
            r2[p] = reach[p] * avg[a]
            v = self._ev(child, r2)
            for j in range(self.n):
                totals[j] += v[j] if j != p else avg[a] * v[j]
        return totals

    def _br(self, node, reach, i):
        """Best-response values for player i vs everyone else's average strategy."""
        if node.terminal:
            return self.terminal_values(node, reach, only=i)[i]
        p = node.player
        if p == i:
            return np.max([self._br(c, reach, i) for c in node.children], axis=0)
        avg = self.avg_strategy(node)
        total = np.zeros(len(self.combos[i]))
        for a, child in enumerate(node.children):
            r2 = list(reach)
            r2[p] = reach[p] * avg[a]
            total += self._br(child, r2, i)
        return total

    def root_reach(self):
        return [np.ones(len(cs)) for cs in self.combos]

    def evs(self):
        v = self._ev(self.root, self.root_reach())
        return [v[i].sum() / self.Z for i in range(self.n)]

    def exploitability(self):
        """Per-player gain from best-responding (chips), and the total."""
        evs = self.evs()
        gaps = [self._br(self.root, self.root_reach(), i).sum() / self.Z - evs[i]
                for i in range(self.n)]
        return evs, gaps

    def solve(self, iters, target_pct, check_every=25, verbose=True):
        t0 = time.time()
        for t in range(1, iters + 1):
            for i in range(self.n):
                self._cfr(self.root, self.root_reach(), t, i)
            self.iterations = t
            if t % check_every == 0 or t == iters:
                evs, gaps = self.exploitability()
                pct = 100 * sum(gaps) / self.cfg.pot
                if verbose:
                    print(f"  iter {t:5d}   exploitability {pct:7.3f}% of pot   "
                          f"({time.time() - t0:.1f}s)")
                if pct <= target_pct:
                    break
        return self.exploitability()


_CATS = ["High card", "Pair", "Two pair", "Trips", "Straight", "Flush",
         "Full house", "Quads", "Straight flush"]


def _describe(s):
    def r(v):
        return RANKS[v - 2]
    cat = s[0]
    if cat == 6:
        return f"Full house {r(s[1])}{r(s[1])}{r(s[1])}{r(s[2])}{r(s[2])}"
    if cat == 7:
        return f"Quads {r(s[1])}s, {r(s[2])} kicker"
    if cat in (4, 8):
        return f"{_CATS[cat]}, {r(s[1])} high"
    return f"{_CATS[cat]} " + "".join(r(v) for v in s[1:])


# --------------------------------------------------------------------------
# Display / tree browser
# --------------------------------------------------------------------------
def show_node(solver, node, reach, path, by_combo=False):
    n = solver.n
    names = solver.names
    print()
    print("=" * 78)
    print("Line: " + (" > ".join(path) if path else "(root)"))
    stacks = [solver.cfg.stack - c for c in node.contribs]
    print(f"Pot {fmt(node.pot)}   |   behind: " +
          ", ".join(f"{names[i]} {fmt(stacks[i])}" for i in range(n) if node.active[i]))
    # range sizes
    sizes = []
    for i in range(n):
        if node.active[i]:
            sizes.append(f"{names[i]} {(reach[i] * solver.weights[i]).sum():.2f} combos")
    print("Ranges here: " + ", ".join(sizes))

    vals = solver._ev(node, reach)
    mass = [solver._contract(solver.P, reach, i) for i in range(n)]

    if node.terminal:
        if node.kind == "fold":
            w = node.active.index(True)
            print(f"Terminal: everyone else folded, {names[w]} wins {fmt(node.pot)}")
        else:
            print("Terminal: showdown between " +
                  ", ".join(names[i] for i in range(n) if node.active[i]))
        for i in range(n):
            tot = (reach[i] * mass[i]).sum()
            if tot > 0:
                print(f"  {names[i]} EV here: {(reach[i] * vals[i]).sum() / tot:.3f}")
        return

    p = node.player
    avg = solver.avg_strategy(node)
    A = len(node.children)
    combos_here = reach[p] * solver.weights[p]
    total = combos_here.sum()
    print(f"\n{names[p]} to act.")
    if total > 0:
        overall = (avg * combos_here).sum(axis=1) / total
        print("Overall: " + "   ".join(f"{node.labels[a]}: {100 * overall[a]:.1f}%" for a in range(A)))
    for i in range(n):
        if i != p and node.active[i]:
            tot = (reach[i] * mass[i]).sum()
            if tot > 0:
                print(f"{names[i]} EV here: {(reach[i] * vals[i]).sum() / tot:.3f}")

    # group rows
    cs = solver.combos[p]
    order = sorted(range(len(cs)), key=lambda k: -solver.strength[p][k])
    groups = {}
    for k in order:
        key = combo_str(cs[k]) if by_combo else hand_class(cs[k])
        groups.setdefault(key, []).append(k)

    short = [l.split(" (")[0] for l in node.labels]
    labw = max(8, max(len(l) for l in short))
    evw = labw + 4
    hdr = f"{'Hand':<7}{'Combos':>7}  " + "".join(f"{l:>{labw + 1}}" for l in short)
    hdr += f"{'EV':>8}   " + "".join(f"{'EV ' + l:>{evw}}" for l in short)
    print()
    print(hdr)
    print("-" * len(hdr))
    for key, ks in groups.items():
        ks = np.array(ks)
        r = reach[p][ks]
        w = combos_here[ks].sum()
        if w <= 1e-9:
            continue
        strat = (avg[:, ks] * combos_here[ks]).sum(axis=1) / w
        m = (r * mass[p][ks]).sum()
        ev = (r * vals[p][ks]).sum() / m if m > 0 else 0.0
        line = f"{key:<7}{w:>7.2f}  " + "".join(f"{100 * s:>{labw}.1f}%" for s in strat)
        line += f"{ev:>8.3f}   "
        for a in range(A):
            line += f"{_action_ev(solver, node, reach, p, a, ks):>{evw}.3f}"
        line += f"   {solver.hand_desc[p][ks[0]]}"
        print(line)


def _action_ev(solver, node, reach, p, a, ks):
    """EV for player p's combos ks if they take action a (others play avg)."""
    child = node.children[a]
    v = solver._ev(child, reach)[p]
    m = solver._contract(solver.P, reach, p)
    r = reach[p][ks]
    den = (r * m[ks]).sum()
    return (r * v[ks]).sum() / den if den > 0 else 0.0


def browse(solver):
    node, reach, path = solver.root, solver.root_reach(), []
    stack = []
    by_combo = False
    while True:
        show_node(solver, node, reach, path, by_combo)
        if not node.terminal:
            print("\nActions: " + "  ".join(f"[{i + 1}] {l}" for i, l in enumerate(node.labels)))
        print("Commands: number = take action, b = back, r = root, c = toggle combos/classes, q = quit")
        try:
            cmd = input("> ").strip().lower()
        except EOFError:
            return
        if cmd == "q":
            return
        if cmd == "c":
            by_combo = not by_combo
        elif cmd == "b" and stack:
            node, reach, path = stack.pop()
        elif cmd == "r":
            node, reach, path, stack = solver.root, solver.root_reach(), [], []
        elif cmd.isdigit() and not node.terminal and 1 <= int(cmd) <= len(node.children):
            a = int(cmd) - 1
            stack.append((node, reach, path))
            avg = solver.avg_strategy(node)
            r2 = list(reach)
            r2[node.player] = reach[node.player] * avg[a]
            path = path + [f"{solver.names[node.player]} {node.labels[a]}"]
            node, reach = node.children[a], r2


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_pcts(s):
    return [float(x) for x in s.replace(" ", "").split(",") if x] if s else []


def ask(prompt, default=None):
    d = f" [{default}]" if default is not None else ""
    v = input(f"{prompt}{d}: ").strip()
    return v if v else (default if default is not None else ask(prompt, default))


def main():
    ap = argparse.ArgumentParser(description="River toy-game solver (DCFR).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--board", help="5 board cards, e.g. AsAdAh2s2d")
    ap.add_argument("--pot", type=float, help="pot at start of river")
    ap.add_argument("--stack", type=float, help="effective stack behind")
    ap.add_argument("--range", action="append", dest="ranges",
                    help="a player's range, in order of action (repeat per player)")
    ap.add_argument("--names", help="comma-separated player names")
    ap.add_argument("--bets", default=None, help="bet sizes, %% of pot (default 50,100)")
    ap.add_argument("--raises", default=None, help="raise sizes, %% of pot after call (default 100)")
    ap.add_argument("--max-raises", type=int, default=None, help="raise cap after a bet (default 1)")
    ap.add_argument("--no-allin", action="store_true", help="don't add an all-in option")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--target", type=float, default=0.1,
                    help="stop at this exploitability, %% of pot (default 0.1)")
    ap.add_argument("--no-browse", action="store_true", help="print the root and exit")
    args = ap.parse_args()

    interactive = args.board is None or not args.ranges
    if interactive:
        print("River toy-game solver — press Enter to accept defaults.\n")
        args.board = args.board or ask("Board (5 cards)", "AsAdAh2s2d")
        args.pot = args.pot if args.pot is not None else float(ask("Pot", "10"))
        args.stack = args.stack if args.stack is not None else float(ask("Effective stack", "10"))
        if not args.ranges:
            n = int(ask("Number of players", "2"))
            defaults = ["KK,33", "88", "QQ"]
            args.ranges = [ask(f"Range for player {i + 1} "
                               f"({'first to act' if i == 0 else 'acts after ' + str(i)})",
                               defaults[i] if i < len(defaults) else None)
                           for i in range(n)]
        if args.bets is None:
            args.bets = ask("Bet sizes (% pot)", "50,100")
        if args.raises is None:
            args.raises = ask("Raise sizes (% pot)", "100")
        if args.max_raises is None:
            args.max_raises = int(ask("Max raises after a bet", "1"))
    args.pot = 10.0 if args.pot is None else args.pot
    args.stack = 10.0 if args.stack is None else args.stack
    args.bets = "50,100" if args.bets is None else args.bets
    args.raises = "100" if args.raises is None else args.raises
    args.max_raises = 1 if args.max_raises is None else args.max_raises

    try:
        board = parse_cards(args.board)
        if len(board) != 5 or len(set(board)) != 5:
            raise ValueError("board must be 5 distinct cards")
        ranges = [parse_range(r, board) for r in args.ranges]
        if len(ranges) < 2:
            raise ValueError("need at least 2 players")
        names = [x.strip() for x in args.names.split(",")] if args.names else None
        if names and len(names) != len(ranges):
            raise ValueError("--names must match number of ranges")
        cfg = TreeConfig(len(ranges), args.pot, args.stack, parse_pcts(args.bets),
                         parse_pcts(args.raises), args.max_raises, not args.no_allin)
        solver = RiverSolver(board, ranges, cfg, names)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    nodes = list(iter_nodes(solver.root))
    print(f"\nBoard {' '.join(card_str(c) for c in board)}   pot {fmt(cfg.pot)}   "
          f"stack {fmt(cfg.stack)}   players {solver.n}")
    for i in range(solver.n):
        print(f"  {solver.names[i]:>4}: {len(solver.combos[i])} combos  ({args.ranges[i]})")
    print(f"Tree: {sum(not x.terminal for x in nodes)} decision nodes, "
          f"{sum(x.terminal for x in nodes)} terminals\n")
    print("Solving...")
    evs, gaps = solver.solve(args.iters, args.target)
    print("\nResult:")
    for i in range(solver.n):
        print(f"  {solver.names[i]:>4} EV {evs[i]:8.3f}  ({100 * evs[i] / cfg.pot:5.1f}% of pot)"
              f"   could gain {gaps[i]:.4f} by deviating")
    if solver.n > 2:
        print("  (multiway: CFR has no convergence guarantee, but a small 'could gain' "
              "means no player can profit much by deviating)")

    if args.no_browse or not sys.stdin.isatty():
        show_node(solver, solver.root, solver.root_reach(), [])
    else:
        browse(solver)


if __name__ == "__main__":
    main()
