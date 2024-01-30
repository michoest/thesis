from src.utils import state_to_number


class Agent:
    def __init__(self, actions, pi):
        self.actions = actions
        self.pi = lambda state, allowed_actions: pi(self, state, allowed_actions)

    def reset(self):
        pass

    def act(self, state, allowed_actions):
        return self.pi(state, allowed_actions)


class DeterministicAgent(Agent):
    def __init__(self, actions, state_action_table):
        self.actions = actions
        self.state_action_table = state_action_table

        def pi(state, allowed_actions):
            return (
                self.state_action_table[state_to_number(state)]
                if self.state_action_table[state_to_number(state)] in allowed_actions
                else -1
            )

        self.pi = pi
