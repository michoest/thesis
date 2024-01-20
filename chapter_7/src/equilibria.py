import numpy as np

from src.nfg import NormalFormGame, GovernedNormalFormGame
from src.utility import QuadraticTwoPlayerUtility
from src.utils import (
    IntervalUnion,
    NoBestResponseFoundException,
    NoEquilibriumFoundException,
)


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
