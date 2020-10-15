import matplotlib.pyplot as plt

import numpy as np

import common
import four_rooms


def rand_policy(s, g, env):
    s = env.reset(s, g)[:2]
    done = False
    traj = [s]
    actions = []
    while not done:
        a = env.action_space.sample()
        actions.append(a)
        s, _, done, _ = env.step(a)
        s = s[:2]
        traj.append(s)

    return np.array(traj), np.array(actions)


def run(env, N=1000):
    random_trajs = []
    random_actions = []
    random_goals = []
    for _ in range(N):
        s, g = env.sample_sg()
        traj, actions = rand_policy(s, g, env)

        random_trajs.append(traj)
        random_actions.append(actions)
        random_goals.append(g)

    return random_trajs, random_actions, random_goals


def plot_rand(env, random_trajs, random_actions):
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    axes = axes.reshape(-1)
    for idx, ax in enumerate(axes):
        common.plot_traj(env, ax, random_trajs[idx])

    plt.savefig('p2_random_trajs.png',
                bbox_inches='tight', pad_inches=0.1, dpi=300)


def test_rand():
    env = four_rooms.make_rooms()
    trajs, actions, goals = run(env)
    new_trajs = []
    for traj, goal in zip(trajs, goals):
        new_trajs.append(np.concatenate([traj, goal.reshape(-1, 2)]))

    plot_rand(env, new_trajs, actions)


if __name__ == '__main__':
    test_rand()