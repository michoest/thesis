import collections
import copy
import math
import random

import numpy as np

from src.nfg import NormalFormGame, GovernedNormalFormGame
from src.utility import QuadraticTwoPlayerUtility


class NoEquilibriumFoundException(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)


class NoBestResponseFoundException(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)


class NoOptimumFoundException(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)


RestrictionSolverResult = collections.namedtuple(
    "RestrictionSolverResult",
    "game optimal_restriction optimal_nash_equilibrium optimal_social_utility "
    "initial_restriction initial_nash_equilibrium initial_social_utility info",
)
RestrictionSolverException = collections.namedtuple(
    "RestrictionSolverException", "game exception args"
)


class IntervalUnion:
    def __init__(self, intervals=[(-np.Inf, np.Inf)]):
        assert isinstance(intervals, list)
        assert all(isinstance(interval, tuple) for interval in intervals)

        self.intervals = intervals

    def __str__(self):
        intervals = (
            " ".join(f"[{a}, {b})" for a, b in self.intervals)
            if self.intervals
            else "()"
        )
        return f"<IntervalUnion {intervals}>"

    def __repr__(self):
        return self.__str__()

    def __bool__(self):
        return bool(self.intervals)

    def __contains__(self, x):
        for a, b in self.intervals:
            if x < a:
                return False
            elif x <= b:
                return True

        return False

    def __len__(self):
        return len(self.intervals)

    def __eq__(self, other):
        for (a, b), (x, y) in zip(self.intervals, other.intervals):
            if a != x or b != y:
                return False

        return True

    def __hash__(self) -> int:
        return hash(tuple(self.intervals))

    def clone(self):
        return IntervalUnion(copy.deepcopy(self.intervals))

    def last_interval_before_or_within(self, x):
        for i, (a, b) in enumerate(self.intervals):
            if x < a:
                return i, (a, b), False
            elif x <= b:
                return i, (a, b), True

        return None, (None, None), False

    def first_interval_after_or_within(self, x):
        for i, (a, b) in reversed(list(enumerate(self.intervals))):
            if x >= b:
                return i, (a, b), False
            elif x >= a:
                return i, (a, b), True

        return None, (None, None), False

    def insert(self, x, y):
        if x >= y:
            return

        i, (a, b), v = self.last_interval_before_or_within(x)
        j, (c, d), w = self.first_interval_after_or_within(y)

        if i is None:
            self.intervals.append((x, y))
        elif j is None:
            self.intervals.insert(0, (x, y))
        else:
            self.intervals[i:j+1] = [
                (x if a is None else min(a, x), y if d is None else max(d, y))
            ]

    def remove(self, x, y):
        if not self.intervals:
            return

        if x is None:
            x = self.intervals[0][0]

        if y is None:
            y = self.intervals[-1][1]

        if x >= y:
            return

        i, (a, b), v = self.last_interval_before_or_within(x)
        j, (c, d), w = self.first_interval_after_or_within(y)

        if i is not None and j is not None:
            if v and (a < x):
                if w and (d > y):
                    self.intervals[i:j+1] = [(a, x), (y, d)]
                else:
                    self.intervals[i:j+1] = [(a, x)]
            else:
                if w:
                    self.intervals[i:j+1] = [(y, d)]
                else:
                    self.intervals[i:j+1] = []

    def clone_and_remove(self, x, y):
        new_interval_union = IntervalUnion(copy.deepcopy(self.intervals))
        new_interval_union.remove(x, y)
        return new_interval_union

    def ndarray(self, step=1.0):
        return np.concatenate([np.arange(a, b, step) for a, b in self.intervals])

    @property
    def complement(self):
        if not self.intervals:
            return IntervalUnion()
        else:
            intervals = (
                [(-np.Inf, self.intervals[0][0])]
                if self.intervals[0][0] != -np.Inf
                else []
            )

            for i in range(1, len(self.intervals)):
                intervals.append((self.intervals[i - 1][1], self.intervals[i][0]))

            if self.intervals[-1][1] != np.Inf:
                intervals.append((self.intervals[-1][1], np.Inf))

            return intervals

    @property
    def inner_complement(self):
        if not self.intervals:
            return IntervalUnion()
        else:
            return [
                (self.intervals[i - 1][1], self.intervals[i][0])
                for i in range(1, len(self.intervals))
            ]

    @property
    def size(self):
        if not self.intervals:
            return 0.0
        elif not self.has_lower_bound() or not self.has_upper_bound():
            return np.inf
        else:
            return sum(b - a for a, b in self.intervals)

    def has_lower_bound(self):
        return (not self.intervals) or (not math.isinf(self.intervals[0][0]))

    def has_upper_bound(self):
        return (not self.intervals) or (not math.isinf(self.intervals[-1][1]))

    def upper_bound(self):
        return None if not self.intervals else self.intervals[-1][1]

    def lower_bound(self):
        return None if not self.intervals else self.intervals[0][0]

    def outer_bounds(self):
        return (
            [] if not self.intervals else [self.intervals[0][0], self.intervals[-1][1]]
        )

    def nearest_elements(self, x):
        if not self.intervals:
            return []

        for i, (a, b) in enumerate(self.intervals):
            if x < a:
                if i > 0:
                    return (
                        [self.intervals[i - 1][1]]
                        if x - self.intervals[i - 1][1] <= a - x
                        else []
                    ) + ([a] if x - self.intervals[i - 1][1] >= a - x else [])
                else:
                    return [a]
            elif x < b:
                return [x]

        return [self.intervals[-1][1]]

    def sample(self):
        assert self.has_lower_bound() and self.has_upper_bound()

        if not self.intervals:
            return None
        else:
            x = random.uniform(0.0, self.size)
            for i, (a, b) in enumerate(self.intervals):
                if x > b - a:
                    x -= b - a
                else:
                    return a + x

        return self.intervals[-1][1]


def is_equilibrium(utilities, x, y, action_space):
    return np.any(
        np.isclose(x, np.array(utilities[0].best_responses(y, action_space)))
    ) and np.any(np.isclose(y, np.array(utilities[1].best_responses(x, action_space))))


def hill_climbing_nash_equilibria(
    game: NormalFormGame,
    action_space: IntervalUnion,
    number_of_samples=10,
    number_of_steps=10,
    decimals=None,
):
    try:
        xs = {action_space.sample() for _ in range(number_of_samples)}
        ps = {
            (x, y)
            for x in xs
            for y in game.utilities[1].best_responses(x, action_space)
        }

        for i in range(number_of_steps):
            ps = {
                (brx, y)
                for (x, y) in ps
                for brx in game.utilities[0].best_responses(y, action_space)
            }
            ps = {
                (x, bry)
                for (x, y) in ps
                for bry in game.utilities[1].best_responses(x, action_space)
            }

        return {
            (round(x, decimals), round(y, decimals)) if decimals is not None else (x, y)
            for (x, y) in ps
            if is_equilibrium(game.utilities, x, y, action_space)
        }
    except NoBestResponseFoundException as e:
        raise NoEquilibriumFoundException(e)


def worst_hill_climbing_nash_equilibrium(
    game: GovernedNormalFormGame, action_space: IntervalUnion, decimals=None
):
    nash_equilibria = hill_climbing_nash_equilibria(
        game, action_space, decimals=decimals
    )
    if not nash_equilibria:
        raise NoEquilibriumFoundException()
    else:
        return min(nash_equilibria, key=game.social_utility)


def quadratic_utility_nash_equilibria(
    game: NormalFormGame, action_space: IntervalUnion, decimals=None
):
    assert all(isinstance(u, QuadraticTwoPlayerUtility) for u in game.utilities)

    u_1, u_2 = game.utilities
    a_1, b_1, c_1, d_1, e_1, f_1 = u_1.coeffs
    a_2, b_2, c_2, d_2, e_2, f_2 = u_2.coeffs

    # Analytical solution is only valid when both utility functions are concave
    # in the respective agents, and if the solution is allowed by action_space.
    # Otherwise, use hill climbing to determine the Nash Equilibria.
    if a_1 < 0 and b_2 < 0:
        x = (c_1 * e_2 - 2 * d_1 * b_2) / (4 * a_1 * b_2 - c_1 * c_2)
        y = (c_2 * d_1 - 2 * e_2 * a_1) / (4 * a_1 * b_2 - c_1 * c_2)

        if x in action_space and y in action_space:
            return {
                (round(x, decimals), round(y, decimals))
                if decimals is not None
                else (x, y)
            }
        else:
            return hill_climbing_nash_equilibria(game, action_space, decimals=decimals)
    else:
        return hill_climbing_nash_equilibria(game, action_space, decimals=decimals)


def worst_quadratic_utility_nash_equilibrium(
    game: GovernedNormalFormGame, action_space: IntervalUnion, decimals=None
):
    nash_equilibria = quadratic_utility_nash_equilibria(
        game, action_space, decimals=decimals
    )
    if not nash_equilibria:
        raise NoEquilibriumFoundException()
    else:
        return min(nash_equilibria, key=game.social_utility)


def absolute_improvement(result):
    return result.optimal_social_utility - result.initial_social_utility


def relative_improvement(result):
    return (
        absolute_improvement(result) / abs(result.initial_social_utility)
        if result.initial_social_utility != 0
        else np.Inf
    )


def degree_of_restriction(result):
    return 1.0 - (result.optimal_restriction.size / result.initial_restriction.size)
