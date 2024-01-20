class NormalFormGame:
    def __init__(self, action_space, utilities):
        self.action_space = action_space
        self.utilities = utilities

    @property
    def number_of_players(self):
        return len(self.utilities)

    def __str__(self):
        return (
            f"<NormalFormGame n={self.number_of_players}, "
            f"A={self.action_space}, u={self.utilities}>"
        )

    def __repr__(self) -> str:
        return self.__str__()


class GovernedNormalFormGame(NormalFormGame):
    def __init__(self, action_space, utilities, social_utility):
        super().__init__(action_space, utilities)
        self.social_utility = social_utility

    def social_optimum(self, action_space=None):
        return self.social_utility.social_optimum(action_space or self.action_space)

    def __str__(self):
        return (
            f"<GovernedNormalFormGame n={self.number_of_players}, "
            f"A={self.action_space}, u={self.utilities}, "
            f"social_utility={self.social_utility}>"
        )


class GovernedNormalFormGameWithOracle(GovernedNormalFormGame):
    def __init__(self, action_space, utilities, social_utility, oracle):
        super().__init__(action_space, utilities, social_utility)
        self.oracle = oracle

    def __str__(self):
        return (
            f"<GovernedNormalFormGameWithOracle n={self.number_of_players}, "
            f"A={self.action_space}, u={self.utilities}, "
            f"social_utility={self.social_utility}>"
        )
