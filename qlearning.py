#! /usr/bin/python
"""
Provides QLearning, a generic implementation of epsilon-greedy classical Q-Learning algorithm for application in RL problems.

Requires an objective formulation of the RL problem to solve i.e. states and actions an agent can be in/execute, as well as user-implemented reward computation.
Iteratively, an action is sampled/selected, which is then executed by the user, who updates the instance with state-action-reward information.
"""

import numpy as np
import random

__author__ = "João André"
__copyright__ = "Copyright 2025, BiRDLab-UMinho"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "João André"
__email__ = "joaocandre@gmail.com"


class QLearning(object):
    """
    Basic implementation of classical Q-Learning for solving generic RL problems.

    @note for generality, no assumption is made on the problem;
          the user is required to
            1) provide a collection of possible states and a collection of actions,
            2) implement the reward function and
            3) iteratively update the instance with state-action-reward information.
    """
    def __init__(self, states, actions, initial_state=None, alpha=0.1, gamma=0.9):
        super(QLearning, self).__init__()

        # learning parameters: learning rate and discount factor
        assert (alpha > 0.0 and alpha < 1.0)
        assert (gamma > 0.0 and gamma < 1.0)
        self._alpha = alpha
        self._gamma = gamma

        # state/action maps
        # @note assume actions are deterministic on all states i.e. same action on same state leads to same reward
        self._states = states
        self._actions = actions

        # current/initial state
        # @note when None, randomize initial state
        if initial_state is None:
            self._current_state_idx = random.randint(0, len(self._states))
        else:
            self._current_state_idx = self.state_index(initial_state)
            assert (self._current_state_idx is not None)

        # initialize learning matrices R (reward) and Q (value)
        self._Rs = np.array([{a: None for a in actions} for _ in range(len(self._states))], dtype=object)
        self._Qs = np.array([{a: 0.0 for a in actions} for _ in range(len(self._states))], dtype=object)

    @property
    def states(self):
        """
        Possible states
        """
        return self._states

    @property
    def actions(self):
        """
        Available actions
        """
        return self._actions

    @property
    def current(self):
        """
        Current state
        """
        return self._states[self._current_state_idx]

    @property
    def R(self):
        """
        Reward matrix, with length matching the number of states, and each element a dictionary indexed on the actions.
        """
        return self._Rs

    @property
    def Q(self):
        """
        Value matrix, with length matching the number of states, and each element a dictionary indexed on the actions.
        """
        return self._Qs

    def state_index(self, state):
        """
        Get state index, from state.

        @note       Requires searching in internal state array. For generality, no structure/order is assumed on states, and linear search is used (O[n] complexity).
                    In theory, this can add overhead on large state arrays (i.e. problems with higher dimensionality).
                    However, through profiling, a linear search was found to be faster when number of states is >10000 (approx).
                    Notwithstanding, one can alternatively sort/structure self._states as an array of tuples (over a S-column array, where S is the number of state variables):
                        self._states = np.ascontiguousarray(self._states).view(dtype=np.dtype([('d' + str(d), sts.dtype) for d, _ in enumerate(self._states[0])])).ravel()
                    and then use binary search for finding state index:
                        np.searchsorted(self._states.flatten(), state)

        :param      state:  State to search.
        :type       state:  self.states.dtype (defined on construction)

        :returns:   Index of given *state* in internal state vector.
        :rtype:     int
        """
        for idx, st in enumerate(self._states):
            if np.all(st == state):
                return idx

        return None

    def get_action(self, epsilon=0.1, min_reward=None):
        """
        Select next action.

        @note       Action selection only; it remains up to the user to execute action and update instance with 'update(state, action, reward)'

        :param      epsilon:     Exploration factor (0: no exploration, 1: randomized action). Defaults to 0.1 (low exploration).
        :type       epsilon:     float
        :param      min_reward:  Minimum reward for action selection. Actions with associated reward in R matrix are ignored.
        :type       min_reward:  float

        :returns:   Action.
        :rtype:     self.actions.dtype (defined on construction)

        :raises     ValueError:  If no valid/selectable actions (> min_reward) exist in current state
        """
        actions = self._actions
        # filter actions with given *min_reward*
        # @note useful to avoid repeatedly explore 'bad' actions
        if min_reward is not None:
            actions = [a for a in self._actions if (self._Rs[self._current_state_idx][a] is None or self._Rs[self._current_state_idx][a] >= min_reward)]

        # raise exception if no valid action is possible
        if not len(actions):
            raise ValueError('No valid actions at state ' + str(self._states[self._current_state_idx]) + ' w/ reward > ' + str(min_reward))

        if np.random.rand() < epsilon:
            # explore new paths/solutions
            idx = random.randint(0, len(actions) - 1)
        else:
            # exploit current information in Q (value) matrix
            idx = np.argmax([self._Qs[self._current_state_idx][a] for a in actions])

        return actions[idx]

    def update(self, new_state, action=None, reward=None):
        """
        Update state, and optionally R (reward) and Q (value) matrices, given state transition, action taken and reward obtained.

        :param      new_state:       New state
        :type       new_state:       self.states.dtype (defined on construction)
        :param      action:          Action taken
        :type       action:          self.actions.dtype (defined on construction)
        :param      reward:          Reward obtained
        :type       reward:          float

        :raises     ValueError:      If *new_state* is not a valid state
        """
        # find state index
        new_idx = self.state_index(new_state)
        if new_idx is None:
            raise ValueError('Invalid state \'' + new_state + '\'')

        # if no action or reward is given, skip learning updates
        # @note useful when fully-exploiting a trained problem, only the current state is updated
        if action is not None and reward is not None:
            # update reward matrix
            # @note given 'reward' argument, R matrix becomes redundant, as it influences the Q-value directly
            self._Rs[self._current_state_idx][action] = reward

            # update Q-value; essentially, a Bellman equation discounting future rewards
            # @note cf. https://en.wikipedia.org/wiki/Q-learning#Algorithm
            self._Qs[self._current_state_idx][action] = (1 - self._alpha) * self._Qs[self._current_state_idx][action] + self._alpha * (reward + self._gamma * max([self._Qs[new_idx][a] for a in self._actions]))

        # update current state
        self._current_state_idx = new_idx
