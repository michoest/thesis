import numpy as np

from src.utils import NoBestResponseFoundException, NoOptimumFoundException


class UtilityFunction:
    def __init__(self, player):
        self.player = player


class QuadraticTwoPlayerUtility(UtilityFunction):
    def __init__(self, player, coeffs):
        super().__init__(player)

        self.coeffs = np.array(coeffs)

    def __call__(self, x):
        return (
            self.coeffs[0] * x[0] ** 2
            + self.coeffs[1] * x[1] ** 2
            + self.coeffs[2] * x[0] * x[1]
            + self.coeffs[3] * x[0]
            + self.coeffs[4] * x[1]
            + self.coeffs[5]
        )

    def best_response(self, x, action_space):
        best_responses = self.best_responses(x, action_space)

        if not best_responses:
            raise NoBestResponseFoundException()
        else:
            return best_responses[0]

    def best_responses(self, x, action_space):
        assert self.player is not None
        assert x in action_space, f"{x} is not allowed in {action_space}"

        if self.player == 0:
            a, b, c, d, e, f = self.coeffs
        elif self.player == 1:
            b, a, c, e, d, f = self.coeffs

        if a == 0:
            if c * x + d == 0:
                # Function is constant in the player: Any response is a best response
                raise NoBestResponseFoundException("Constant function!")
            elif c * x + d > 0:
                # Function is linear with positive slope: Maximum is upper bound if it
                # exists
                if not action_space.has_upper_bound():
                    raise NoBestResponseFoundException()
                else:
                    return [action_space.upper_bound()]
            else:
                # Function is linear with negative slope: Maximum is lower bound if it
                # exists
                if not action_space.has_lower_bound():
                    raise NoBestResponseFoundException()
                else:
                    return [action_space.lower_bound()]
        elif a > 0:
            # Function is convex in the player: Maximum is one of the outer bounds
            if not action_space.has_lower_bound() or not action_space.has_upper_bound():
                raise NoBestResponseFoundException()
            else:
                candidates = action_space.outer_bounds()
        else:
            # Function is concave in the player: Maximum is in the middle or close to it
            candidates = action_space.nearest_elements((c * x + d) / (-2 * a))

        candidate_values = [self((c, x)) for c in candidates]
        maximum_value = max(candidate_values)

        return [x for x, y in zip(candidates, candidate_values) if y == maximum_value]

    def social_optimum(self, action_space):
        assert (
            self.player is None
        ), "The social optimum is only defined for a social utility function"

        a, b, c, d, e, f = self.coeffs
        if 4 * a * b == c**2:
            raise NoOptimumFoundException()

        x, y = (c * e - 2 * b * d) / (4 * a * b - c**2), (c * d - 2 * a * e) / (
            4 * a * b - c**2
        )

        if x in action_space and y in action_space:
            return self((x, y))
        else:
            raise NotImplementedError()

    def __add__(self, other):
        return QuadraticTwoPlayerUtility(
            None if self.player != other.player else self.player,
            self.coeffs + other.coeffs,
        )

    def __str__(self):
        return (
            f"<QuadraticTwoPlayerUtility {self.coeffs[0]}x^2 + {self.coeffs[1]}y^2 + "
            f"{self.coeffs[2]}xy + {self.coeffs[3]}x + {self.coeffs[4]}y + "
            f"{self.coeffs[5]}>"
        )

    def __repr__(self) -> str:
        return self.__str__()
