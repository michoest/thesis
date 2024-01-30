from src.utils import delta


class Environment:
    def __init__(self, initial_state, delta):
        self.state = initial_state
        self.initial_state = initial_state
        self.delta = delta

    def reset(self):
        self.state = self.initial_state

    def move(self, joint_action):
        self.state = self.delta(self.state, joint_action)


class BinaryAttributeEnvironment(Environment):
    def __init__(self, initial_state):
        super().__init__(initial_state, delta)
