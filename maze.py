#! /usr/bin/python
"""
Provides Maze and Agent classes, that generate a random 2D maze of variable size, and a 2D agent that travels within the maze.

Maze wraps around *mazelib*, and randomly assigns a goal position within the maze.
Each Agent instance is constructed for a specific Maze instance, and can move in orthogonal directions (left, right, up, down).
"""

import mazelib
from mazelib.generate.Prims import Prims

import random
import numpy as np
import matplotlib.pyplot as plt

# force matplotlib to respond to Ctrl+C
# cf. https://stackoverflow.com/a/75864329/2340002
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

__author__ = "João André"
__copyright__ = "Copyright 2025, BiRDLab-UMinho"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "João André"
__email__ = "joaocandre@gmail.com"


class Maze(object):
    """
    Simple class implementing a 2D maze, of (internal) size *rows* x *cols*
    @note external border/wall is added to given maze size, and an odd number (closest to given shape) of rows/cols is used (hardoced in the  maze generator module)

    """
    def __init__(self, rows, cols, goal=None):
        super(Maze, self).__init__()
        self._m = mazelib.Maze()
        self._m.generator = Prims(int(0.5 * rows), int(0.5 * cols))
        self._m.generate()
        self.reset(goal)

    def reset(self, goal=None):
        """
        Resets the target goal coordinates of the maze with given *goal*, or randomly selects a maze position as goal.

        :param      goal: Goal coordinates as (row, col)
        :type       goal: tuple, list, numpy.ndarray
        """
        self._goal = [random.randint(0, self.size[0]), random.randint(0, self.size[1])]
        while not self.is_valid(self._goal):
            self._goal = [random.randint(0, self.size[0]), random.randint(0, self.size[1])]

    @property
    def size(self):
        """
        Get maze size

        :returns:   Maze size as (rows, cols)
        :rtype:     tuple
        """
        return self._m.grid.shape

    @property
    def goal(self):
        """
        Get goal state

        :returns:   Goal position as (row, col)
        :rtype:     tuple
        """
        return self._goal

    @property
    def valid(self):
        """
        Get all valid states

        :returns:   Goal position as (row, col)
        :rtype:     tuple
        """
        return np.argwhere(self._m.grid == 0)

    @property
    def all(self):
        """
        Get all states

        :returns:   Goal position as (row, col)
        :rtype:     tuple
        """
        return np.argwhere(self._m.grid != None)

    def is_valid(self, pos):
        """
        Check if given *pos* is a valid maze position

        :param      pos:  Position/coordinates as (row, col)
        :type       pos:  tuple, list, numpy.ndarray

        :returns:   True if *pos* on open path in maze, False otherwise.
        :rtype:     bool
        """
        try:
            return self._m.grid[pos[0], pos[1]] == 0
        except IndexError:
            return False

    def is_goal(self, pos):
        """
        Check if given *pos* is the target goal state of the maze

        :param      pos:  Position/coordinates as (row, col)
        :type       pos:  tuple, list, numpy.ndarray

        :returns:   True if *pos* on open path in the target goal state, False otherwise
        :rtype:     bool
        """
        try:
            return pos == self._goal
        except IndexError:
            return False

    def plot(self, show=True):
        """
        Plot maze grid as black/white matrix (black: wall, white: path)

        :param      show:  The show
        :type       show:  bool

        :returns:   { description_of_the_return_value }
        :rtype:     { return_type_description }
        """
        fig = plt.figure(figsize=(10, 5))
        mat = plt.imshow(self._m.grid, cmap=plt.cm.binary, interpolation='nearest')
        plt.scatter(self._goal[1], self._goal[0], marker='x', c='tab:red', s=100)
        plt.xticks([]), plt.yticks([])
        if show:
            plt.show()

        return fig, mat

    def create_agent(self, pos=None):
        """
        Creates an agent to travel the 2D maze.

        :param      pos:  Initial position of the agent in the maze,  as (row, col).
                          If None, random position is used.
        :type       pos:  tuple, list, numpy.ndarray

        :returns:   Agent instance.
        :rtype:     Agent
        """
        return Agent(self, pos)


class Agent(object):
    """
    Class implementing an active agent in a 2D maze
    """
    def __init__(self, maze, pos=None):
        super(Agent, self).__init__()
        self._maze = maze
        self.reset(pos)

    @property
    def pos(self):
        return self._pos

    def reset(self, pos=None):
        if pos is not None:
            # clip and assign position
            self._pos = [np.clip(pos[0], 0, self._maze.size[0]), np.clip(pos[1], 0, self._maze.size[1])]
        else:
            # randomize position
            self._pos = [random.randint(0, self._maze.size[0]), random.randint(0, self._maze.size[1])]
            while not self._maze.is_valid(self._pos):
                self._pos = [random.randint(0, self._maze.size[0]), random.randint(0, self._maze.size[1])]

    def n_actions(self):
        # hardcoded
        # @todo remove this, not needed
        return 4

    def left(self):
        self._pos[1] -= 1
        if self._maze.is_valid(self._pos):
            return True
        else:
            self._pos[1] += 1

        return False

    def right(self):
        self._pos[1] += 1
        if self._maze.is_valid(self._pos):
            return True
        else:
            self._pos[1] -= 1

        return False

    def up(self):
        self._pos[0] -= 1
        if self._maze.is_valid(self._pos):
            return True
        else:
            self._pos[0] += 1

        return False

    def down(self):
        self._pos[0] += 1
        if self._maze.is_valid(self._pos):
            return True
        else:
            self._pos[0] -= 1

        return False

    def random(self):
        """
        Execute random action.

        :returns:   Action name (i.e. one of ['left', 'right', 'up', 'down']) and success/failure (i.e. if next state is valid)
        :rtype:     str, bool
        """
        actions = ['left', 'right', 'up', 'down']
        n = random.randint(0, len(actions) - 1)
        if n == 0:
            return 'left', self.left()
        elif n == 1:
            return 'right', self.right()
        elif n == 2:
            return 'up', self.up()
        elif n == 3:
            return 'down', self.down()

    def move(self, action=None):
        """
        Execute action, verbose overload

        :param      action:  Action to execute, one of ['left', 'right', 'up', 'down']
        :type       action:  str

        :returns:   True if action is valid (next position is not a maze wall)
        :rtype:     bool
        """
        # execute action
        if action == 'left':
            return self.left()
        elif action == 'right':
            return self.right()
        elif action == 'up':
            return self.up()
        elif action == 'down':
            return self.down()

        return self.random()

    def plot(self, action=None, show=True):
        """
        Display agent position in the maze grid

        :param      show:  Action highlight. Influences marker used for plotting agent position.
        :type       show:  str
        :param      show:  Whether to show the maze (blocking call)
        :type       show:  bool
        """
        fig, mat = self._maze.plot(show=False)
        # select marker
        marker = 'o'
        if action == 'left':
            marker = 8
        elif action == 'right':
            marker = 9
        elif action == 'up':
            marker = 10
        elif action == 'down':
            marker = 11

        plt.scatter(self._pos[1], self._pos[0], marker=marker, c='k', s=100, alpha=0.5)

        if show:
            plt.show()

        return fig, mat
