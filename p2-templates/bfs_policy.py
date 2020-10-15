
from queue import Queue
import matplotlib.pyplot as plt

import numpy as np

import common
import four_rooms


def bfs_policy(s, g, env):
    visited = set()
    q = Queue()
    q.put(s)
    prev = np.full((*env.map.shape, 2), -1)
    prev[s[0], s[1], :] = 0
    while not q.empty():
        c = q.get()
        if (c[0], c[1]) in visited:
            continue

        if np.all(np.equal(c, g)):
            traj = [c]
            actions = []
            while True:
                new_c = prev[c[0], c[1], :]
                motion = c - new_c
                traj.append(new_c)
                actions.append(env.act_map[(motion[0], motion[1])])
                c = new_c
                if np.all(np.equal(c, s)):
                    break

            # import pdb; pdb.set_trace()

            return np.array(traj[::-1]), np.array(actions[::-1])

        next_states = np.add(env.act_set, c)
        next_states = next_states[env.map[next_states[:, 0], next_states[:, 1]]]
        for ns in next_states:
            if not (ns[0], ns[1]) in visited:
                prev[ns[0], ns[1], :] = c
                q.put(ns)

        visited.add((c[0], c[1]))

    raise RuntimeError('Trajectory not found.')


def run(env, N=1000):
    expert_trajs = []
    expert_actions = []
    for _ in range(N):
        s, g = env.sample_sg()
        traj, actions = bfs_policy(s, g, env)

        expert_trajs.append(traj)
        expert_actions.append(actions)

    return expert_trajs, expert_actions


def plot_bfs(env, expert_trajs, expert_actions):
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    axes = axes.reshape(-1)
    for idx, ax in enumerate(axes):
        common.plot_traj(env, ax, expert_trajs[idx])

    plt.savefig('p2_expert_trajs.png',
                bbox_inches='tight', pad_inches=0.1, dpi=300)


def test_bfs():
    env = four_rooms.make_rooms()
    trajs, actions = run(env)
    plot_bfs(env, trajs, actions)


if __name__ == '__main__':
    test_bfs()