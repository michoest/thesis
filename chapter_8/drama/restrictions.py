# Typing
from typing import Optional, Set, Callable, Any, Union

# Standard modules
import math
from decimal import Decimal, getcontext
from abc import ABC
import random

# External modules
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box


class Restriction(ABC, gym.Space):
    """Base class for restrictions. All restrictions are valid :class:`gymnasium.spaces.Space`'s."""
    def __init__(
        self,
        base_space: gym.Space,
        *,
        seed: int | np.random.Generator | None = None,
    ):
        """Constructor of :class:`Restriction`.

        Args:
            base_space: :class:`gymnasium.spaces.Space` whose subsets can be represented by the restriction.
            seed: Random seed for sampling. Defaults to None.
        """
        super().__init__(base_space.shape, base_space.dtype, seed)
        self.base_space = base_space

    def __repr__(self) -> str:
        """Representation of the Restriction."""
        return f"{self.__class__.__name__}"


class DiscreteRestriction(Restriction, ABC):
    """Representation of a :class:`gymnasium.spaces.Discrete` restriction."""
    def __init__(
        self,
        base_space: gym.spaces.Discrete,
        *,
        seed: int | np.random.Generator | None = None,
    ):
        """Constructor of :class:`DiscreteRestriction`.

        Args:
            base_space: :class:`gymnasium.spaces.Discrete` whose subsets can be represented by the
            restriction.
            seed: Random seed for sampling. Defaults to None.
        """
        super().__init__(base_space, seed=seed)


class ContinuousRestriction(Restriction, ABC):
    """Representation of a :class:`gymnasium.spaces.Box` restriction."""
    def __init__(
        self,
        base_space: gym.spaces.Box,
        *,
        seed: int | np.random.Generator | None = None,
    ):
        """Constructor of :class:`ContinuousRestriction`.

        Args:
            base_space: :class:`gymnasium.spaces.Box` whose subsets can be represented by the restriction.
            seed: Random seed for sampling. Defaults to None.
        """
        super().__init__(base_space, seed=seed)


class DiscreteSetRestriction(DiscreteRestriction):
    """Representation of a :class:`gymnasium.spaces.Discrete` restriction  as a set of allowed actions."""
    def __init__(
        self,
        base_space: gym.spaces.Discrete,
        *,
        allowed_actions: Optional[Set[int]] = None,
        seed: int | np.random.Generator | None = None,
    ):
        """Constructor of :class:`DiscreteSetRestriction`.

        Args:
            base_space: :class:`gymnasium.spaces.Discrete` whose subsets can be represented by the restriction
            allowed_actions: Optional, initial set of allowed actions
            seed: Random seed for sampling. Defaults to None
        """
        super().__init__(base_space, seed=seed)

        self.allowed_actions = (
            allowed_actions
            if allowed_actions is not None
            else set(range(base_space.start, base_space.start + base_space.n))
        )

    @property
    def is_np_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`gymnasium.spaces.Box`.

        Returns:
            `True`
        """
        return True

    def sample(self, mask: Any | None = None) -> int:
        """Randomly sample an action from the allowed set.

        Args:
            mask: The mask used for sampling (currently no effect)

        Returns:
            Valid discrete action
        """
        return random.choice(tuple(self.allowed_actions))

    def contains(self, x: int) -> bool:
        """Check if a discrete action is allowed.

        Args:
            x: The discrete action

        Returns:
            `True` if the action is allowed. `False` otherwise
        """
        return x in self.allowed_actions

    def add(self, x: int) -> None:
        """Adds a discrete action to the set of allowed actions.

        Args:
            x: The discrete action
        """
        self.allowed_actions.add(x)

    def remove(self, x: int) -> None:
        """Removes a discrete action from the set of allowed actions.

        Args:
            x: The discrete action
        """
        self.allowed_actions.remove(x)

    def __eq__(self, __value: object) -> bool:
        """Check if two instances of :class:`DiscreteSetRestriction` are equal."""
        return (
            isinstance(__value, DiscreteSetRestriction)
            and self.base_space == __value.base_space
            and self.allowed_actions == __value.allowed_actions
        )

    def __repr__(self) -> str:
        """Representation of the :class:`DiscreteSetRestriction`."""
        return f"{self.__class__.__name__}({self.allowed_actions})"


class DiscreteVectorRestriction(DiscreteRestriction):
    """Representation of a :class:`gymnasium.spaces.Discrete` restriction as a boolean vector of allowed
    and forbidden actions.
    """
    def __init__(
        self,
        base_space: gym.spaces.Discrete,
        *,
        allowed_actions: Optional[np.ndarray[bool]] = None,
        seed: int | np.random.Generator | None = None,
    ):
        """Constructor of :class:`DiscreteVectorRestriction`.

        Args:
            base_space: :class:`gymnasium.spaces.Discrete` whose subsets can be represented by the restriction
            allowed_actions: Optional, initial binary vector indicating allowed actions
            seed: Random seed for sampling. Defaults to None
        """
        super().__init__(base_space, seed=seed)

        self.allowed_actions = (
            allowed_actions
            if allowed_actions is not None
            else np.ones(base_space.n, dtype=np.bool_)
        )

    @property
    def is_np_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`gymnasium.spaces.Box`.

        Returns:
            `True`
        """
        return True

    def sample(self, mask: Any | None = None) -> int:
        """Randomly sample an action from the allowed set.

        Args:
            mask: The mask used for sampling (currently no effect)

        Returns:
            Valid discrete action
        """
        return self.base_space.start + random.choice(
            tuple(index for index, value in enumerate(self.allowed_actions) if value)
        )

    def contains(self, x: int) -> bool:
        """Check if a discrete action is allowed.

        Args:
            x: The discrete action

        Returns:
            `True` if the action is allowed. `False` otherwise
        """
        return self.allowed_actions[x - self.base_space.start]

    def __repr__(self) -> str:
        """Representation of the :class:`DiscreteVectorRestriction`."""
        return f"{self.__class__.__name__}({self.allowed_actions})"


class Node(object):
    """Node in the AVL tree of intervals.
    A single instance of :class:`Node` represents an allowed interval.
    """
    def __init__(
        self,
        x: float = None,
        y: float = None,
        left: object = None,
        right: object = None,
        height: int = 1,
    ):
        """Constructor of :class:`Node`.

        Args:
            x: Lower bound of the interval
            y: Upper bound of the interval
            left: Left, smaller interval
            right: Right, larger interval
        """
        self.x: Decimal = Decimal(f"{x}") if x is not None else None
        self.y: Decimal = Decimal(f"{y}") if y is not None else None
        self.left = left
        self.right = right
        self.height = height

    def __str__(self):
        """String representation of the :class:`Node`."""
        return f"<Node ({self.x},{self.y}), height: {self.height}, left: {self.left}, \
        right: {self.right}>"

    def __repr__(self):
        """Representation of the :class:`Node`."""
        return self.__str__()


class IntervalUnionRestriction(ContinuousRestriction):
    """Representation of a one-dimensional :class:`gymnasium.spaces.Box` restriction as an AVL tree of
    allowed intervals."""
    root_tree = None
    size: Decimal = 0
    draw = None

    def __init__(self, base_space: Box):
        """Constructor of :class:`IntervalUnionRestriction`.

        Args:
            base_space: :class:`gymnasium.spaces.Box` whose subsets can be represented by the restriction
        """
        super().__init__(base_space)
        getcontext().prec = 28

        self.root_tree = Node(base_space.low[0], base_space.high[0])
        self.size = Decimal(f"{base_space.high[0]}") - Decimal(f"{base_space.low[0]}")

    @property
    def is_np_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`gymnasium.spaces.Box`.

        Returns:
            `True`
        """
        return True

    def __contains__(self, item):
        """Check if a continuous action is allowed.

        Args:
            item: The continuous action

        Returns:
            `True` if the action is allowed. `False` otherwise
        """
        return self.contains(item)

    def contains(self, x: Union[np.array, float], root: object = "root"):
        """Check if a continuous action is allowed.

        Args:
            x: The continuous action
            root: :class:`Node` to start the search from or 'root' for searching the whole tree, default is 'root'

        Returns:
            `True` if the action is allowed. `False` otherwise
        """
        if root == "root":
            root = self.root_tree

        if isinstance(x, np.ndarray):
            x = x.item()

        x = Decimal(f"{x}")

        if not root:
            return False
        elif root.x <= x <= root.y:
            return True
        elif root.x > x:
            return self.contains(x, root.left)
        else:
            return self.contains(x, root.right)

    def nearest_elements(self, x, root: Node = "root"):
        """Finds the closest allowed actions for a continuous action.

        Args:
            x: The continuous action
            root: :class:`Node` to start the search from or 'root' for searching the whole tree, default is 'root'

        Returns:
            Nearest elements in the action space. Returns x if it is valid.
        """
        if root == "root":
            root = self.root_tree

        x = Decimal(f"{x}")

        if root and x > root.y:
            return self._nearest_elements(x, x - root.y, root.y, root.right)
        elif root and x < root.x:
            return self._nearest_elements(x, root.x - x, root.x, root.left)
        else:
            return [x]

    def _nearest_elements(self, x, min_diff, min_value, root: Node = "root"):
        """Helper function to find the closest allowed actions for a continuous action.

        Args:
            x: The continuous action
            min_diff: Minimum distance of an allowed action to x that has been found so far
            min_value: Allowed action with the minimum distance to x that has been found so far
            root: :class:`Node` to start the search from or 'root' for searching the whole tree, default is 'root'

        Returns:
            Nearest elements in the allowed action space. Returns x if it is valid.
        """
        if root == "root":
            root = self.root_tree

        x = Decimal(f"{x}")
        min_diff = Decimal(f"{min_diff}")
        min_value = Decimal(f"{min_value}")

        if not root:
            return [min_value]
        elif x > root.y:
            distance = x - root.y
            return (
                [min_value, root.y]
                if distance == min_diff
                else [min_value]
                if distance > min_diff
                else self._nearest_elements(x, distance, root.y, root.right)
            )
        elif x < root.x:
            distance = root.x - x
            return (
                [min_value, root.x]
                if distance == min_diff
                else [min_value]
                if distance > min_diff
                else self._nearest_elements(x, distance, root.x, root.left)
            )
        else:
            return [x]

    def nearest_element(self, x, root: Node = "root"):
        """Finds the minimum closest allowed action for a continuous action.

        Args:
            x: The continuous action
            root: :class:`Node` to start the search from or 'root' for searching the whole tree, default is 'root'

        Returns:
            Nearest element in the allowed action space. Returns x if it is valid.
        """
        if root == "root":
            root = self.root_tree

        x = Decimal(f"{x}")

        return self.nearest_elements(x, root)[-1]

    def last_interval_before_or_within(self, x, root: Node = "root"):
        """The last interval before or within a continuous action

        Args:
            x: The continuous action
            root: :class:`Node` to start the search from or 'root' for searching the whole tree, default is 'root'

        Returns:
            Tuple containing the lower and upper boundaries of the interval and a
            variable indicating if the number lies in the interval.

            For example: (root.x, root.y), True
        """
        if root == "root":
            root = self.root_tree

        x = Decimal(f"{x}")

        if root.x <= x <= root.y:
            return (root.x, root.y), True
        elif x < root.x:
            return (
                self.last_interval_before_or_within(x, root.left)
                if root.left is not None
                else ((None, None), False)
            )
        else:
            if root.right is not None:
                interval, flag = self.last_interval_before_or_within(x, root.right)
                if interval[0] is None:
                    interval, flag = (root.x, root.y), False
            else:
                interval, flag = (root.x, root.y), False

            return (
                (interval, flag)
                if root.right is not None
                else ((root.x, root.y), False)
            )

    def first_interval_after_or_within(self, x, root: Node = "root"):
        """The last interval after or within a continuous action

        Args:
            x: The continuous action
            root: :class:`Node` to start the search from or 'root' for searching the whole tree, default is 'root'

        Returns:
            Tuple containing the lower and upper boundaries of the interval and a
            variable indicating if the number lies in the interval.

            For example: (root.x, root.y), True
        """
        if root == "root":
            root = self.root_tree

        x = Decimal(f"{x}")

        if root.x <= x <= root.y:
            return (root.x, root.y), True
        elif x > root.y:
            return (
                self.first_interval_after_or_within(x, root.right)
                if root.right is not None
                else ((None, None), False)
            )
        else:
            if root.left is not None:
                interval, flag = self.first_interval_after_or_within(x, root.left)
                if interval[0] is None:
                    interval, flag = (root.x, root.y), False
            else:
                interval, flag = (root.x, root.y), False

            return (
                (interval, flag) if root.left is not None else ((root.x, root.y), False)
            )

    def smallest_interval(self, root: Node = "root"):
        """Return the Node of the smallest interval

        Args:
            root: :class:`Node` to start the search from or 'root' for searching the whole tree, default is 'root'

        Returns:
            :class:`Node` of the smallest interval
        """
        if root == "root":
            root = self.root_tree

        if root is None or root.left is None:
            return root
        else:
            return self.smallest_interval(root.left)

    def add(self, x, y, root: Node = "root"):
        """Add an interval to the action space

        Args:
            x: Lower bound of the interval
            y: Upper bound of the interval
            root: :class:`Node` to start the insertion from or 'root' for inserting over the whole tree, default is 'root'

        Returns:
            Updated root :class:`Node` of the action space
        """
        assert y > x, "Upper must be larger than lower bound"

        if root == "root":
            root = self.root_tree
            if root is None:
                self.root_tree = Node(x, y)
                self.size += y - x
                return self.root_tree

        x = Decimal(f"{x}")
        y = Decimal(f"{y}")
        if not root:
            self.size += y - x
            return Node(x, y)
        elif y < root.x:
            root.left = self.add(x, y, root.left)
        elif x > root.y:
            root.right = self.add(x, y, root.right)
        else:
            old_size = root.y - root.x
            root.x = min(root.x, x)
            root.y = max(root.y, y)
            self.size += root.y - root.x - old_size

            updated = False
            if root.right is not None and root.y >= root.right.x:
                self.size -= root.y - root.right.y
                root.y = root.right.y
                updated = True

            if root.left is not None and root.x <= root.left.y:
                self.size -= root.left.x - root.x
                root.x = root.left.x
                updated = True

            root.right = self.remove(root.x, root.y, root.right)
            root.left = self.remove(root.x, root.y, root.left)
            if updated:
                root = self.add(x, y, root)

        root.height = 1 + max(self.getHeight(root.left), self.getHeight(root.right))

        b = self.getBal(root)

        if b > 1 and y < root.left.x and self.getBal(root.left) > 0:
            self.root_tree = self.rRotate(root)
            return self.root_tree

        if b < -1 and x > root.right.y and self.getBal(root.right) < 0:
            self.root_tree = self.lRotate(root)
            return self.root_tree

        if b > 1 and x > root.left.y and self.getBal(root.left) < 0:
            root.left = self.lRotate(root.left)
            self.root_tree = self.rRotate(root)
            return self.root_tree

        if b < -1 and y < root.right.x and self.getBal(root.right) > 0:
            root.right = self.rRotate(root.right)
            self.root_tree = self.lRotate(root)
            return self.root_tree

        self.root_tree = root
        return root

    def sample(self, root: Node = "root") -> np.ndarray:
        """Randomly sample a continuous action from a uniform distribution over the allowed action space

        Args:
            root: :class:`Node` node of the action space, default is 'root'

        Returns:
            Sampled continuous action
        """
        if root == "root":
            root = self.root_tree

        if root is None:
            # raise Exception("Empty Action Space") or
            return self.base_space.sample()

        if self.draw is None:
            self.draw = Decimal(f"{random.uniform(0.0, float(self.size))}")

        self.draw -= root.y - root.x
        if self.draw > 0:
            result = None
            if root.left is not None:
                result = self.sample(root.left)
            if not result and root.right is not None:
                result = self.sample(root.right)
            return result
        else:
            result = float(root.y + self.draw)
            self.draw = None
            return np.array([result], dtype=np.float32)

    def remove(self, x, y, root: Node = "root", adjust_size: bool = True):
        """Removes an interval from the action space

        Args:
            x: Lower bound of the interval
            y: Upper bound of the interval
            root: :class:`Node` to start the removal from or 'root' for removing over the whole tree, default is 'root'
            adjust_size: Whether the size attribute of the tree should be modified

        Returns:
            Updated root :class:`Node` of the action space
        """
        assert y > x, "Upper must be larger than lower bound"

        if root == "root":
            root = self.root_tree
            if root is None:
                return root

        x = Decimal(f"{x}")
        y = Decimal(f"{y}")

        if not root:
            return None
        elif x > root.x and y < root.y:
            self.size -= root.y - x
            old_maximum = root.y
            root.y = x
            root = self.add(y, old_maximum, root)
        elif x == root.x and y < root.y:
            self.size -= y - x
            root.x = y
        elif x > root.x and y == root.y:
            self.size -= y - x
            root.y = x
        elif x < root.x < y < root.y:
            self.size -= y - root.x
            root.x = y
            root.left = self.remove(x, y, root.left, adjust_size)
        elif root.x < x < root.y < y:
            self.size -= root.y - x
            root.y = x
            root.right = self.remove(x, y, root.right, adjust_size)
        elif y <= root.x:
            root.left = self.remove(x, y, root.left, adjust_size)
        elif x >= root.y:
            root.right = self.remove(x, y, root.right, adjust_size)
        else:
            if adjust_size:
                self.size -= root.y - root.x
            if root.left is None:
                self.root_tree = self.remove(x, y, root.right, adjust_size)
                return self.root_tree
            elif root.right is None:
                self.root_tree = self.remove(x, y, root.left, adjust_size)
                return self.root_tree
            rgt = self.smallest_interval(root.right)
            root.x = rgt.x
            root.y = rgt.y
            root.right = self.remove(rgt.x, rgt.y, root.right, adjust_size=False)
            root = self.remove(x, y, root, adjust_size)
        if not root:
            return None

        root.height = 1 + max(self.getHeight(root.left), self.getHeight(root.right))

        b = self.getBal(root)

        if b > 1 and self.getBal(root.left) > 0:
            self.root_tree = self.rRotate(root)
            return self.root_tree

        if b < -1 and self.getBal(root.right) < 0:
            self.root_tree = self.lRotate(root)
            return self.root_tree

        if b > 1 and self.getBal(root.left) < 0:
            root.left = self.lRotate(root.left)
            self.root_tree = self.rRotate(root)
            return self.root_tree

        if b < -1 and self.getBal(root.right) > 0:
            root.right = self.rRotate(root.right)
            self.root_tree = self.lRotate(root)
            return self.root_tree

        self.root_tree = root
        return root

    def lRotate(self, z: Node):
        """Performs a left rotation. Switches roles of parent and child nodes.

        Args:
            z: Parent :class:`Node` for the rotation

        Returns:
            Updated parent :class:`Node`
        """
        y = z.right
        T2 = y.left

        y.left = z
        z.right = T2

        z.height = 1 + max(self.getHeight(z.left), self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left), self.getHeight(y.right))

        return y

    def rRotate(self, z: Node):
        """Performs a right rotation. Switches roles of parent and child nodes.

        Args:
            z: Parent :class:`Node` for the rotation

        Returns:
            Updated parent :class:`Node`
        """
        y = z.left
        T3 = y.right

        y.right = z
        z.left = T3

        z.height = 1 + max(self.getHeight(z.left), self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left), self.getHeight(y.right))

        return y

    def getHeight(self, root: Node = "root"):
        """Returns the height of a Node

        Args:
            root: :class:`Node` to return the height from or 'root' for the height of the whole tree, default is 'root'

        Returns:
            The height of the node in the tree
        """
        if root == "root":
            root = self.root_tree

        if not root:
            return 0

        return root.height

    def getBal(self, root: Node = "root"):
        """Calculate the balance factor

        Args:
            root: :class:`Node` to calculate the balance factor for or 'root' for the balance factor of the whole tree,
                default is 'root'

        Returns:
            The balance factor
        """
        if root == "root":
            root = self.root_tree

        if not root:
            return 0

        return self.getHeight(root.left) - self.getHeight(root.right)

    def intervals(self):
        """Return all intervals of the allowed action space in an ordered way.

        Returns:
            List of tuples containing the ordered intervals.

            For example: [(0.1,0.5), (0.7,0.9)]
        """
        return self._intervals()

    def _intervals(self, root: Node = "root"):
        """Return all allowed intervals starting from a specific node in an ordered way.

        Args:
            root: :class:`Node` to start the search from or 'root' for searching the whole tree, default is 'root'

        Returns:
            List of tuples containing the ordered intervals.

            For example: [(0.1,0.5), (0.7,0.9)]
        """
        if root == "root":
            root = self.root_tree

        if root is None:
            return []

        ordered = []
        if root.left is not None:
            ordered = ordered + self._intervals(root.left)
        ordered.append((float(root.x), float(root.y)))
        if root.right is not None:
            ordered = ordered + self._intervals(root.right)
        return ordered

    def __str__(self):
        """String representation of the :class:`IntervalUnionRestriction`."""
        return f"{self.__class__.__name__}({self.intervals()})"

    def __repr__(self):
        """Representation of the :class:`IntervalUnionRestriction`."""
        return self.__str__()


class BucketSpaceRestriction(ContinuousRestriction):
    """Representation of a one-dimensional :class:`gymnasium.spaces.Box` restriction as a binary vector
    indicating the availability of equally sized buckets."""
    def __init__(
        self,
        base_space: Box,
        bucket_width=1.0,
        epsilon=0.01,
        available_buckets: np.ndarray = None,
    ) -> None:
        """Constructor of :class:`BucketSpaceRestriction`.

        Args:
            base_space: :class:`gymnasium.spaces.Box` whose subsets can be represented by the restriction
            bucket_width: The width of each bucket
            epsilon: The radius in which buckets are set valid/invalid around a specific point
            available_buckets: The binary vector indicating the allowed subsets
        """
        super().__init__(base_space)
        assert isinstance(self.base_space, Box)

        self.a, self.b = Decimal(f"{self.base_space.low.item()}"), Decimal(
            f"{self.base_space.high.item()}"
        )
        self.bucket_width, self.epsilon = Decimal(f"{bucket_width}"), Decimal(
            f"{epsilon}"
        )
        self.number_of_buckets = math.ceil((self.b - self.a) / self.bucket_width)

        if available_buckets:
            assert (
                len(available_buckets) == self.number_of_buckets
            ), "Not all available bucket indicators provided!"
            assert np.all(
                [index in [1.0, 0.0] for index in available_buckets]
            ), "No boolean bucket indicators!"
            self.buckets = available_buckets
        else:
            self.buckets = np.ones((self.number_of_buckets,), dtype=bool)

    @property
    def is_np_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`gymnasium.spaces.Box`.

        Returns:
            `True`
        """
        return True

    def contains(self, x):
        """Check if a continuous action is allowed.

        Args:
            x: The continuous action

        Returns:
            `True` if the action is allowed. `False` otherwise
        """
        return False if x < self.a or x >= self.b else self.buckets[self._bucket(x)]

    def sample(self, mask: None = None):
        """Randomly sample a continuous action from a uniform distribution over the allowed action space

        Args:
            mask: The mask used for sampling (currently no effect)

        Returns:
            Sampled continuous action
        """
        if not self.intervals:
            return None
        else:
            x = Decimal(f"{random.uniform(0.0, float(self.b - self.a))}")

            for i, (a, b) in enumerate(self.intervals):
                if x > Decimal(b) - Decimal(a):
                    x -= Decimal(b) - Decimal(a)
                else:
                    return Decimal(a) + x

        return self.intervals[-1][1]

    def clone(self):
        """Returns a copy of the :class:`BucketSpaceRestriction`

        Returns:
            :class:`BucketSpaceRestriction` copy
        """
        assert isinstance(self.base_space, Box)

        return BucketSpaceRestriction(
            self.base_space,
            bucket_width=float(self.bucket_width),
            epsilon=float(self.epsilon),
            available_buckets=self.buckets,
        )

    def clone_and_remove(self, x):
        """Returns a copy of the :class:`BucketSpaceRestriction` without buckets containing a specific value

        Args:
            x: Buckets containing this value should be removed from the allowed action space

        Returns:
            :class:`BucketSpaceRestriction` copy
        """
        space = self.clone()
        space.remove(x)
        return space

    def remove(self, x, with_epsilon=True):
        """Remove buckets containing a specific value from the allowed action space

        Args:
            x: Buckets containing this value should be removed from the allowed action space
            with_epsilon: If `True`, a subset of epsilon around x is removed.
                Otherwise, only buckets containing the specific value are removed.
        """
        x = Decimal(f"{x}")

        if with_epsilon:
            self._set(x, False)
        else:
            self.buckets[self._bucket(x)] = False

    def add(self, x, with_epsilon=True):
        """Add buckets containing a specific value to the allowed action space

        Args:
            x: Buckets containing this value should be added to the allowed action space
            with_epsilon: If `True`, a subset of epsilon around x is added.
                Otherwise, only buckets containing the specific value are added.
        """
        x = Decimal(f"{x}")

        if with_epsilon:
            self._set(x)
        else:
            self.buckets[self._bucket(x)] = True

    @property
    def intervals(self):
        """Return all intervals of the allowed action space in an ordered way.

        Returns:
            List of tuples containing the ordered intervals.

            For example: [(0.1,0.5), (0.7,0.9)]
        """
        a, intervals = None, []
        for i in range(self.number_of_buckets):
            if a is None:
                if self.buckets[i]:
                    a = self.a + i * self.bucket_width
            elif not self.buckets[i]:
                intervals.append((float(a), float(self.a + i * self.bucket_width)))
                a = None
            elif i == self.number_of_buckets - 1:
                intervals.append((float(a), float(self.b)))

        return intervals

    def _bucket(self, x):
        """Return the bucket which contains a specific value

        Args:
            x: Value for which the bucket has to be found

        Returns:
            Indicator of the bucket
        """
        return math.floor((x - self.a) / self.bucket_width)

    def _set(self, x, value=True):
        """Set the indicator value for the bucket of a specific value manually.

        Args:
            x: The indicator value for the bucket containing x is modified
            value: If `True`, the bucket containing x belongs to the allowed action space.
                Otherwise, the bucket is unavailable.
        """
        lower_bucket = (
            self._bucket(x - self.epsilon) if x - self.epsilon >= self.a else None
        )
        upper_bucket = (
            self._bucket(x + self.epsilon) if x + self.epsilon <= self.b else None
        )

        if lower_bucket is None:
            if upper_bucket is None:
                self.buckets = (
                    np.ones((self.number_of_buckets,), dtype=bool)
                    if value
                    else np.zeros((self.number_of_buckets,), dtype=bool)
                )
            else:
                self.buckets[: upper_bucket + 1] = value
        else:
            if upper_bucket is None:
                self.buckets[lower_bucket:] = value
            else:
                self.buckets[lower_bucket : upper_bucket + 1] = value

    def reset(self):
        """Resets the action space to the unrestricted state"""
        self.buckets = np.ones((self.number_of_buckets,), dtype=bool)

    def __str__(self):
        """String representation of the :class:`IntervalUnionRestriction`."""
        intervals = (
            " ".join(f"[{float(a)}, {float(b)})" for a, b in self.intervals)
            if self.intervals
            else "()"
        )
        return f"<BucketSpace {intervals}>"

    def __repr__(self):
        """Representation of the :class:`IntervalUnionRestriction`."""
        return self.__str__()

    def __bool__(self):
        return bool(np.any(self.buckets))

    def __contains__(self, item):
        return self.contains(item)

    def __hash__(self):
        return hash((self.a, self.b, self.bucket_width, tuple(self.intervals)))

    def __eq__(self, other):
        return (self.a, self.b, self.bucket_width, tuple(self.intervals)) == (
            other.a,
            other.b,
            other.bucket_width,
            tuple(other.intervals),
        )


class PredicateRestriction(Restriction):
    """Representation of an arbitrary space as the set of elements for which a predicate is True."""
    def __init__(
        self,
        base_space: gym.Space,
        *,
        predicate: Optional[Callable[[Any], bool]] = None,
        seed: int | np.random.Generator | None = None,
    ):
        super().__init__(base_space, seed=seed)

        self.predicate = predicate if predicate is not None else (lambda x: True)

    @property
    def is_np_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`gymnasium.spaces.Box`.

        Returns:
            `False`
        """
        return False

    def sample(self, mask: Any | None = None) -> int:
        """Randomly sample a set of elements for which the predicate is True

        Args:
            mask: The mask used for sampling (currently no effect)

        Returns:
            Sampled set of elements
        """
        raise NotImplementedError

    def contains(self, x: Any) -> bool:
        """Check if an action is allowed and the predicate is True.

        Args:
            x: The action

        Returns:
            `True` if the action is allowed. `False` otherwise
        """
        return self.base_space.contains(x) and self.predicate(x)
