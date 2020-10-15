"""F20 10-703 HW2
# 10-703: Homework 2 Part 1-Behavior Cloning & DAGGER

You will implement this assignment in this python file

You are given helper functions to plot all the required graphs
"""
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt
import random
import time

import gym
import numpy as np

from imitation import Imitation


plt.style.use('seaborn')


MODE_BC = 'behavior cloning'
MODE_DAGGER = 'dagger'


def generate_imitation_results(mode, expert_file, keys=[100],
                               num_seeds=1, num_iterations=100):
    # Number of training iterations. Use a small number
    # (e.g., 10) for debugging, and then try a larger number
    # (e.g., 100).

    # Dictionary mapping number of expert trajectories to a list of rewards.
    # Each is the result of running with a different random seed.
    reward_data = OrderedDict({key: [] for key in keys})
    accuracy_data = OrderedDict({key: [] for key in keys})
    loss_data = OrderedDict({key: [] for key in keys})

    for num_episodes in keys:
        for t in range(num_seeds):
            print('*' * 50)
            print('num_episodes: %s; seed: %d' % (num_episodes, t))

            # Create the environment.
            env = gym.make('CartPole-v0')
            env.seed(t)  # set seed
            im = Imitation(env, num_episodes, expert_file)
            print("Evaluating Expert")
            expert_reward = im.evaluate(im.expert)
            print('Expert reward: %.2f' % expert_reward)

            loss_vec = []
            acc_vec = []
            imitation_reward_vec = []
            for i in range(num_iterations):
                if mode == MODE_BC:
                    im.generate_behavior_cloning_data()
                elif mode == MODE_DAGGER:
                    im.generate_dagger_data()
                else:
                    raise RuntimeError(f'Mode "{mode}" is not supported.')

                # WRITE CODE HERE
                loss, acc = im.train()

                print("Evaluating Imitation")
                imitation_reward = im.evaluate(im.model)

                print('Imitation Reward: %.2f' % imitation_reward)

                loss_vec.append(loss)
                acc_vec.append(acc)
                imitation_reward_vec.append(imitation_reward)

            if t == 0:
                reward_data[num_episodes] = imitation_reward_vec
                accuracy_data[num_episodes] = acc_vec
                loss_data[num_episodes] = loss_vec
            else:
                reward_data[num_episodes] += 1 / (t + 1) * (np.subtract(imitation_reward_vec, reward_data[num_episodes]))
                accuracy_data[num_episodes] += 1 / (t + 1) * (np.subtract(acc_vec, accuracy_data[num_episodes]))
                loss_data[num_episodes] += 1 / (t + 1) * (np.subtract(loss_vec, loss_data[num_episodes]))
    # END

    return reward_data, accuracy_data, loss_data, expert_reward


"""### Experiment: Student vs Expert
In the next two cells, you will compare the performance of the expert policy
to the imitation policies obtained via behavior cloning and DAGGER.
"""


def plot_student_vs_expert(mode, expert_file, keys=[100],
                           num_seeds=1, num_iterations=100):
    assert len(keys) == 1
    reward_data, acc_data, loss_data, expert_reward = \
        generate_imitation_results(mode, expert_file, keys,
                                   num_seeds, num_iterations)

    # Plot the results
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 5))
    # WRITE CODE HERE
    x_vals = range(len(reward_data[keys[0]]))

    ax1.axhline(y=expert_reward, color='r')
    ax1.text(0, expert_reward + 5, "Expert Reward", color='r')
    ax1.set_ylim(100, expert_reward + 20)

    ax1.plot(x_vals, reward_data[keys[0]], label="Reward")
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    ax1.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)

    ax2.plot(x_vals, acc_data[keys[0]], label="Accuracy")
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)

    ax3.plot(x_vals, loss_data[keys[0]], label="Loss")
    ax3.set_ylabel('Loss')
    ax3.set_xlabel('Iteration')
    ax3.grid(True)

    # END
    plt.savefig('p1_student_vs_expert_%s.png' % mode, dpi=300)
    # plt.show()


def plot_compare_num_episodes(mode, expert_file, keys, num_seeds=1, num_iterations=100):
    """Plot the reward, loss, and accuracy for each, remembering to label each line.
    """
    s0 = time.time()
    reward_data, accuracy_data, loss_data, expert_reward = \
        generate_imitation_results(mode, expert_file, keys, num_seeds, num_iterations)

    ### Plot the results
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 5))

    ax1.axhline(y=expert_reward, color='r')
    ax1.text(0, expert_reward + 5, "Expert Reward", color='r')
    ax1.set_ylim(100, expert_reward + 20)

    # WRITE CODE HERE
    lines = []
    for key in keys:
        x_vals = range(len(reward_data[key]))
        label = f"{key}"

        ax1.plot(x_vals, reward_data[key], label=label)
        line = ax2.plot(x_vals, accuracy_data[key], label=label)[0]
        ax3.plot(x_vals, loss_data[key], label=label)

        lines.append(line)

    ax1.set_ylabel('Reward')
    ax1.grid(True)
    ax1.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    # ax1.legend(title='Num Expert Episodes')

    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    # ax2.legend(title='Num Expert Episodes')

    ax3.set_ylabel('Loss')
    ax3.set_xlabel('Iteration')
    ax3.grid(True)
    # ax3.legend(title='Num Expert Episodes')
    # fig.legend(lines, labels=keys, loc="center right", title='Num Expert Episodes')
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", title='Num Expert Episodes')
    plt.subplots_adjust(right=0.85)
    # END
    plt.savefig('p1_expert_data_%s.png' % mode, dpi=300)
    # plt.show()
    print('time cost', time.time() - s0)


def main(args=None):
    # generate all plots for Problem 1
    expert_file = 'expert.h5'

    # switch mode
    if args is not None:
        mode = args.mode.lower()
        keys = args.keys
        num_seeds = args.seeds
        num_seeds = args.seeds
        num_iterations = args.iters
    else:
        mode = MODE_BC
        # mode = MODE_DAGGER

        # change the list of num_episodes below for testing and different tasks
        keys = [100]  # [1, 10, 50, 100]
        num_seeds = 3  # 3
        num_iterations = 100  # Number of training iterations. Use a small number
                              # (e.g., 10) for debugging, and then try a larger number
                              # (e.g., 100).

    # Q1.1.1, Q1.2.1
    # plot_student_vs_expert(mode, expert_file, keys, num_seeds=num_seeds, num_iterations=num_iterations)

    # Q1.1.2, Q1.2.2
    plot_compare_num_episodes(mode, expert_file, keys, num_seeds=num_seeds, num_iterations=num_iterations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, choices=[MODE_BC, MODE_DAGGER], default=MODE_BC)
    parser.add_argument('-k', '--keys', type=int, nargs='+', default=[100])
    parser.add_argument('-s', '--seeds', type=int, default=3)
    parser.add_argument('-i', '--iters', type=int, default=100)

    main(parser.parse_args())
