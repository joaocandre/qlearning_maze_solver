#! /usr/bin/python
"""
Simple script implementing dynamic maze generation and solving with a classical Q-Learning approach.

On each call, a new maze is generated (using the Maze class wrapping around mazelib),
and a simple epsilon-greedy Q-Learning implementation is used to train an agent to solve it.
An exploitative rollout is then conducted where the agent solves the maze from a randomized initial position,
and the resulting path is plotted for visual inspection.

"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from maze import Maze, Agent
from qlearning import QLearning

__author__ = "João André"
__copyright__ = "Copyright 2025, BiRDLab-UMinho"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "João André"
__email__ = "joaocandre@gmail.com"


def reward(maze, agent, valid):
    """
    Computes the reward value on a maze-solving RL problem

    :param      maze:   Maze object.
    :type       maze:   Maze
    :param      agent:  Agent (maze traveller)
    :type       agent:  Agent
    :param      valid:  Wether the last action take was valid.
    :type       valid:  bool

    :returns:   1 if goal state has been reached, 0 if a valid action was taken, -1 otherwise.
    :rtype:     int
    """
    if valid:
        if maze.is_goal(agent.pos):
            return 1
        return 0

    return -1


def solve(maze, agent, ql, epsilon=0.1, interactive=False):
    """
    Solve a maze using a (trained/untrained) Q (value) matrix.

    :param      maze:        2D Maze to solve.
    :type       maze:        Maze
    :param      agent:       2D Agent (maze traveler)
    :type       agent:       Agent
    :param      ql:          Q-Learning instance
    :type       ql:          QLearning
    :param      epsilon:     Exploration factor (0: no exploration, 1: randomized action). Defaults to 0.1.
    :type       epsilon:     float
    :param      interactive: Whether to display agent progress towards goal state. Defaults to False.
    :type       interactive: bool

    :returns:   Path of the agent towards the goal, as a collection of intermediary states
    :rtype:     numpy.ndarray
    """
    # randomize agent position & update Q-Learning state/values
    agent.reset()
    ql.update(agent.pos)

    # move agent until goal state is reached
    path = np.array([agent.pos])
    while not maze.is_goal(agent.pos):
        # get next action
        # @note min_reward is relevant for efficient exploration, by avoiding low-reward state-action pairs (e.g. moving towards a maze wall)
        action = ql.get_action(epsilon=epsilon, min_reward=0.0)
        # plot agent state and selected action
        if interactive:
            print(ql.current, action)
            agent.plot(action=action)
        # execute action
        valid = agent.move(action)
        # compute reward
        r = reward(maze, agent, valid)
        # update Q-Learning state/values
        # @note advisable to update after training if epsilon > 0.0 (fostering exploration w/ epsilon-greedy approach)
        ql.update(agent.pos, action, reward=r)
        # backup agent state
        path = np.vstack([path, agent.pos])

    return path


def train(maze, agent, alpha=0.1, gamma=0.9, max_iter=100, epsilon=0.8):
    """
    Train an 2D agent to solve a 2D maze using classical Q-Learning.
    A total of *max_iter* are conducted with high exploration

    :param      maze:      2D Maze to solve.
    :type       maze:      Maze
    :param      agent:     2D Agent (maze traveler)
    :type       agent:     Agent
    :param      alpha:     Learning rate. Defaults to 0.1.
    :type       alpha:     float
    :param      gamma:     Discount factor. Defaults to 0.9.
    :type       gamma:     float
    :param      max_iter:  Maximum number of iterations. Defaults to 100.
    :type       max_iter:  int
    :param      epsilon:   Exploration factor (0: no exploration, 1: randomized action). Defaults to 0.8.
    :type       epsilon:   float

    :returns:   Trained QLearning instance.
    :rtype:     QLearning
    """
    ql = QLearning(states=maze.valid,
                   actions=['left', 'right', 'up', 'down'],
                   initial_state=agent.pos,
                   alpha=alpha, gamma=gamma)

    # iterative learning loop
    # @note maze is solved in each episode (from a random initial state until the goal state)
    for _ in tqdm(range(max_iter), desc='Training'):
        solve(maze, agent, ql, epsilon=epsilon)

    return ql


################################################################################################################################################

if __name__ == "__main__":
    # initialize maze and agent (maze traveler)
    maze = Maze(10, 10)
    agent = maze.create_agent()  # alternative, use 'agent = Agent(maze)'
    maze.plot()

    # train agent
    # @note epsilon = 0.8 balances high exploration w/ faster iterations (generally orienting agent towards goal)
    ql = train(maze, agent, max_iter=200, epsilon=0.8)

    # solve maze with trained agent
    # @note epsilon = 0.0 ensures max. exploitation
    path = solve(maze, agent, ql, epsilon=0.0)

    # display agent path until goal
    maze.plot(show=False)
    plt.scatter(path[0][1], path[0][0], marker='o', c='b', s=100, alpha=0.5)
    for p in path[1:]:
        plt.scatter(p[1], p[0], marker='.', c='b', s=100, alpha=0.5)
    plt.show()
