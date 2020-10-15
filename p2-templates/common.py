import numpy as np


def action_to_one_hot(env, action):
    action_vec = np.zeros(env.action_space.n)
    action_vec[action] = 1
    return action_vec


def plot_traj(env, ax, traj, goal=None):
    traj_map = env.map.copy().astype(np.float)
    traj_map[traj[:, 0], traj[:, 1]] = 2 # visited states
    traj_map[traj[0, 0], traj[0, 1]] = 1.5 # starting state
    traj_map[traj[-1, 0], traj[-1, 1]] = 2.5 # ending state
    if goal is not None:
        traj_map[goal[0], goal[1]] = 3 # goal
    ax.imshow(traj_map)
    ax.set_xlabel('y')
    ax.set_label('x')
