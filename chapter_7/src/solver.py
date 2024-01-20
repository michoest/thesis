import math
from collections import deque

import numpy as np

from src.nfg import (
    NormalFormGame,
    GovernedNormalFormGame,
    GovernedNormalFormGameWithOracle,
)
from src.utils import (
    RestrictionSolverResult,
    NoEquilibriumFoundException,
    IntervalUnion,
)


class IntervalUnionRestrictionSolver:
    def __init__(
        self,
        *,
        epsilon=0.1,
        decimals=None,
        timeout_steps=None,
        timeout_explored_restrictions=None
    ):
        assert epsilon > 0

        self.epsilon = epsilon
        self.decimals = decimals or math.ceil(-math.log(self.epsilon, 10))
        self.timeout_steps = timeout_steps
        self.timeout_explored_restrictions = timeout_explored_restrictions

    def solve(
        self, game: GovernedNormalFormGame, *, nash_equilibrium_oracle=None
    ) -> RestrictionSolverResult:
        # Decide which oracle function to use
        if nash_equilibrium_oracle is None:
            if isinstance(game, GovernedNormalFormGameWithOracle):
                nash_equilibrium_oracle = game.oracle
            else:
                nash_equilibrium_oracle = self._nash_equilibrium_oracle

        # Keep track of explored restrictions to avoid double work
        explored_restrictions, current_step = set(), 0

        # Initialize optimum with current restriction (i.e., full action_space)
        try:
            (
                initial_restriction,
                initial_equilibrium,
            ) = game.action_space, nash_equilibrium_oracle(
                game, game.action_space, decimals=self.decimals
            )
            explored_restrictions.add(initial_restriction)

            optimal_restriction, optimal_social_utility = initial_restriction, np.round(
                game.social_utility(initial_equilibrium), decimals=self.decimals
            )

            # Maintain a queue with all open (unexplored) restrictions
            restriction_queue = deque([(initial_restriction, initial_equilibrium)])
            while restriction_queue:
                current_restriction, current_equilibrium = restriction_queue.pop()

                for relevant_action in self._relevant_actions(current_equilibrium):
                    current_step += 1

                    new_restriction = current_restriction.clone_and_remove(
                        round(relevant_action - self.epsilon, self.decimals),
                        round(relevant_action + self.epsilon, self.decimals),
                    )

                    if new_restriction and not (
                        new_restriction in explored_restrictions
                    ):
                        explored_restrictions.add(new_restriction)

                        try:
                            new_equilibrium = nash_equilibrium_oracle(
                                game, new_restriction, decimals=self.decimals
                            )
                            restriction_queue.append((new_restriction, new_equilibrium))

                            # Update optimum if new_restriction is better
                            new_social_utility = np.round(
                                game.social_utility(new_equilibrium),
                                decimals=self.decimals,
                            )
                            if (new_social_utility > optimal_social_utility) or (
                                new_social_utility == optimal_social_utility
                                and new_restriction.size > optimal_restriction.size
                            ):
                                optimal_restriction, optimal_social_utility = (
                                    new_restriction,
                                    new_social_utility,
                                )

                        except NoEquilibriumFoundException:
                            # New restriction does not have an equilibrium, so we
                            # cannot use it for further restrictions
                            continue

                # Check if one of the timeout conditions is met
                if (
                    self.timeout_steps is not None
                    and current_step >= self.timeout_steps
                ) or (
                    self.timeout_explored_restrictions is not None
                    and len(explored_restrictions) >= self.timeout_explored_restrictions
                ):
                    break
        except NoEquilibriumFoundException as e:
            raise e
        else:
            optimal_equilibrium = nash_equilibrium_oracle(
                game, optimal_restriction, decimals=self.decimals
            )
            initial_social_utility = np.round(
                game.social_utility(initial_equilibrium), decimals=self.decimals
            )

            return RestrictionSolverResult(
                game,
                optimal_restriction,
                optimal_equilibrium,
                optimal_social_utility,
                initial_restriction,
                initial_equilibrium,
                initial_social_utility,
                {"number_of_oracle_calls": len(explored_restrictions)},
            )

    # Generic solver for restricted Nash Equilibrium (only used if no specialized
    # solver is available)
    def _nash_equilibrium_oracle(
        game: NormalFormGame, restriction: IntervalUnion
    ) -> tuple:
        raise NotImplementedError()

    def _relevant_actions(self, joint_action):
        # joint_action can either be one joint action or a list of joint actions
        return (
            set(sum(joint_action, ()))
            if isinstance(joint_action, list)
            else set(joint_action)
        )
