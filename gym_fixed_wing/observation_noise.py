import numpy as np
from gym.utils import seeding


class OrnsteinUhlenbeckNoise:
    """
    A Ornstein Uhlenbeck action noise, this is designed to aproximate brownian motion with friction.

    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

    :param mean: (float) the mean of the noise
    :param sigma: (float) the scale of the noise
    :param theta: (float) the rate of mean reversion
    :param dt: (float) the timestep for the noise
    :param initial_noise: ([float]) the initial value for the noise output, (if None: 0)
    """

    def __init__(self, mean, sigma, theta=.15, dt=1e-2, initial_noise=None):
        self._theta = theta
        self._mu = mean
        self._sigma = sigma
        self._dt = dt
        self.initial_noise = initial_noise
        self.noise_prev = None
        self.np_random = np.random.RandomState()
        self.reset()

    def __call__(self):
        noise = self.noise_prev + self._theta * (self._mu - self.noise_prev) * self._dt + \
                self._sigma * np.sqrt(self._dt) * self.np_random.normal(size=self._mu.shape)
        self.noise_prev = noise
        return noise

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def reset(self):
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev = self.initial_noise if self.initial_noise is not None else self.np_random.normal(size=self._mu.shape) * self._sigma

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self._mu, self._sigma)