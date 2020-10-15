import argparse
import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import uniform_filter

import gcbc
import four_rooms
import rand_policy
import bfs_policy


plt.style.use('seaborn')

MODE_VANILLA = 'v'
MODE_RELABEL = 'r'


def main(args):
    env = four_rooms.make_rooms()

    if args.random:
        trajs, actions, random_goals = rand_policy.run(env)
    else:
        trajs, actions = bfs_policy.run(env)
        random_goals = None

    gcbc_agent = gcbc.GCBC(env, trajs, actions, random_goals)

    if args.mode == MODE_VANILLA:
        gcbc_agent.generate_behavior_cloning_data()
    else:
        gcbc_agent.generate_relabel_data()

    loss_vecs = []
    acc_vecs = []
    succ_vecs = []
    for i in range(args.seeds):
        print('*' * 50)
        print('seed: %d' % i)
        loss_vec = []
        acc_vec = []
        succ_vec = []
        gcbc_agent.reset_model()

        for e in range(args.iterations):
            loss, acc = gcbc_agent.train(num_epochs=20)
            succ = gcbc.evaluate_gc(env, gcbc_agent)
            loss_vec.append(loss)
            acc_vec.append(acc)
            succ_vec.append(succ)
            print(e, round(loss, 3), round(acc, 3), succ)
        loss_vecs.append(loss_vec)
        acc_vecs.append(acc_vec)
        succ_vecs.append(succ_vec)

    loss_vec = np.mean(np.array(loss_vecs), axis=0).tolist()
    acc_vec = np.mean(np.array(acc_vecs), axis=0).tolist()
    succ_vec = np.mean(np.array(succ_vecs), axis=0).tolist()

    ### Plot the results
    # you may use uniform_filter(succ_vec, 5) to smooth succ_vec
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 7))

    ax1.set_title('{} Policy, {} Relabeling'.format(
        "Random" if args.random else "Expert",
        "With" if args.mode == MODE_RELABEL else "Without"),
        size=18)

    # plt.figure(figsize=(12, 4))
    # WRITE CODE HERE
    x_vals = range(args.iterations)
    ax1.plot(x_vals, loss_vec, label='Training Loss')
    ax1.set_ylabel('Training Loss')
    ax1.grid(True)
    ax1.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)

    ax2.plot(x_vals, acc_vec, label='Training Accuracy')
    ax2.set_ylabel('Training Accuracy')
    ax2.grid(True)
    ax2.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)

    ax3.plot(x_vals, uniform_filter(succ_vec), label='Success Rate')
    ax3.set_ylabel('Success Rate')
    ax3.set_ylim(0, 1)
    ax3.grid(True)
    ax3.set_xlabel('Iteration')


    # fig.legend(loc="center right")
    # plt.subplots_adjust(right=0.85)
    # END
    plt.savefig('p2_gcbc_random_%s_%s.png' % (args.random, args.mode), dpi=300)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--random', action='store_true')
    parser.add_argument('-m', '--mode', type=str, choices=[MODE_VANILLA, MODE_RELABEL])
    parser.add_argument('-s', '--seeds', type=int, default=3)
    parser.add_argument('-i', '--iterations', type=int, default=200)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    main(args)
