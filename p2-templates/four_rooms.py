import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

import common


def make_rooms(l=5, T=30):
    # build env
    env = FourRooms(l, T)
    # Visualize the map
    env.render_map()

    return env


def test_rooms(l=5):
    env = make_rooms()

    s = np.array([1, 1])
    g = np.array([2*l+1, 2*l+1])
    s = env.reset(s, g)[:2]
    done = False
    traj = [s]
    while not done:
        a = env.action_space.sample()
        s, _, done, _ = env.step(a)
        s = s[:2]
        traj.append(s)
    traj = np.array(traj)

    ax = plt.subplot()
    common.plot_traj(env, ax, traj, g)
    plt.savefig('p2_random_traj.png',
                bbox_inches='tight', pad_inches=0.1, dpi=300)
    # plt.show()


class FourRooms:
    SUCCESS = 'succ'
    FAIL = 'fail'

    def __init__(self, l=5, T=30):
        '''
        FourRooms Environment for pedagogic purposes
        Each room is a l*l square gridworld,
        connected by four narrow corridors,
        the center is at (l+1, l+1).
        There are two kinds of walls:
        - borders: x = 0 and 2*l+2 and y = 0 and 2*l+2
        - central walls
        T: maximum horizion of one episode
            should be larger than O(4*l)
        '''
        assert l % 2 == 1 and l >= 5
        self.l = l
        self.total_l = 2 * l + 3
        self.T = T

        # create a map: zeros (walls) and ones (valid grids)
        self.map = np.ones((self.total_l, self.total_l), dtype=np.bool)
        # build walls
        self.map[0, :] = self.map[-1, :] = self.map[:, 0] = self.map[:, -1] = False
        self.map[l+1, [1,2,-3,-2]] = self.map[[1,2,-3,-2], l+1] = False
        self.map[l+1, l+1] = False

        # define action mapping (go right/up/left/down, counter-clockwise)
        # e.g [1, 0] means + 1 in x coordinate, no change in y coordinate hence
        # hence resulting in moving right
        self.act_set = np.array([
            [1, 0], [0, 1], [-1, 0], [0, -1]
        ], dtype=np.int)
        self.action_space = spaces.Discrete(4)

        # you may use self.act_map in search algorithm
        self.act_map = {
                (1, 0): 0,
              (0, 1): 1,
              (-1, 0): 2,
              (0, -1): 3
        }
        # self.act_map[(1, 0)] = 0
        # self.act_map[(0, 1)] = 1
        # self.act_map[(-1, 0)] = 2
        # self.act_map[(0, -1)] = 3

    def render_map(self):
        # plt.imshow(self.map)
        plt.xlabel('y')
        plt.ylabel('x')
        plt.savefig('p2_map.png',
                    bbox_inches='tight', pad_inches=0.1, dpi=300)
        # plt.show()

    def sample_sg(self):
        """ Generate a random state and goal on the map. The state and goal will:
                - Not be in a wall
                - Not be the same position.
        """
        # sample s
        while True:
            s = [np.random.randint(self.total_l),
                np.random.randint(self.total_l)]
            if self.map[s[0], s[1]]:
                break

        # sample g
        while True:
            g = [np.random.randint(self.total_l),
                np.random.randint(self.total_l)]
            if self.map[g[0], g[1]] and \
                (s[0] != g[0] or s[1] != g[1]):
                break

        return np.array(s), np.array(g)

    def reset(self, s=None, g=None):
        '''
        s np.array((2,)): Optional starting position, if ommited will be randomly
            generated
        g np.array((2,)): Optional goal, if ommited will be randomly
            generated

        return obs: np.cat(s, g)
        '''
        if s is None or g is None:
            s, g = self.sample_sg()
        else:
            assert 0 < s[0] < self.total_l - 1 and 0 < s[1] < self.total_l - 1
            assert 0 < g[0] < self.total_l - 1 and 0 < g[1] < self.total_l - 1
            assert (s[0] != g[0] or s[1] != g[1])
            assert self.map[s[0], s[1]] and self.map[g[0], g[1]]

        self.s = s
        self.g = g
        self.t = 1

        return self._obs()

    def step(self, a):
        '''
        a: action, a scalar
        return obs, reward, done, info
        - done: whether the state has reached the goal
        - info: succ if the state has reached the goal, fail otherwise
        '''
        assert self.action_space.contains(a)

        # WRITE CODE HERE
        attempted_s = np.add(self.s, self.act_set[a])
        if self.map[attempted_s[0], attempted_s[1]]:
            self.s = attempted_s
        self.t += 1

        at_goal = np.all(np.equal(self.g, self.s))
        done = at_goal or (self.t == self.T)
        info = self.SUCCESS if at_goal else self.FAIL
        # END

        return self._obs(), 0.0, done, info

    def _obs(self):
        obs = np.concatenate([self.s, self.g])
        assert len(obs) == 4
        return obs


if __name__ == '__main__':
    test_rooms()