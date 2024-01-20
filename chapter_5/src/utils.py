import numpy as np


def state_to_number(state):
    return np.sum(state * (2 ** np.flip(np.arange(len(state)))))


def delta(state, actions):
    # Change a value if it is included in actions
    return np.array(
        [not value if index in actions else value for index, value in enumerate(state)]
    )
