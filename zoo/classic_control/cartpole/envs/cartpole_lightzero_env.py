import copy
from datetime import datetime
from typing import Union, Optional, Dict

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs import ObsPlusPrevActRewWrapper
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

class RunningMeanStd(object):
    """
    Overview:
       The RunningMeanStd class is a utility that maintains a running mean and standard deviation calculation over
        a stream of data.
    Interfaces:
        __init__, update, reset, mean, std
    Properties:
        - mean (:obj:`np.ndarray`): The running mean.
        - std (:obj:`np.ndarray`): The running standard deviation.
        - _epsilon (:obj:`float`): A small number to prevent division by zero when calculating standard deviation.
        - _shape (:obj:`tuple`): The shape of the data stream.
        - _mean (:obj:`np.ndarray`): The current mean of the data stream.
        - _var (:obj:`np.ndarray`): The current variance of the data stream.
        - _count (:obj:`float`): The number of data points processed.
    """

    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        """
        Overview:
            Initialize the RunningMeanStd object.
        Arguments:
            - epsilon (:obj:`float`, optional): A small number to prevent division by zero when calculating standard
                deviation. Default is 1e-4.
            - shape (:obj:`tuple`, optional): The shape of the data stream. Default is an empty tuple, which
                corresponds to scalars.
        """
        self._epsilon = epsilon
        self._shape = shape
        self.reset()

    def update(self, x: np.array):
        """
        Overview:
            Update the running statistics with a new batch of data.
        Arguments:
            - x (:obj:`np.array`): A batch of data.
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        new_count = batch_count + self._count
        mean_delta = batch_mean - self._mean
        new_mean = self._mean + mean_delta * batch_count / new_count
        # this method for calculating new variable might be numerically unstable
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(mean_delta) * self._count * batch_count / new_count
        new_var = m2 / new_count
        self._mean = new_mean
        self._var = new_var
        self._count = new_count

    def reset(self):
        """
        Overview:
            Resets the state of the environment and reset properties:  \
                ``_mean``, ``_var``, ``_count``
        """
        self._mean = np.zeros(self._shape, 'float64')
        self._var = np.ones(self._shape, 'float64')
        self._count = self._epsilon

    @property
    def mean(self) -> np.ndarray:
        """
        Overview:
            Get the current running mean.
        Returns:
            The current running mean.
        """
        return self._mean

    @property
    def std(self) -> np.ndarray:
        """
        Overview:
            Get the current running standard deviation.
        Returns:
            The current running mean.
        """
        return np.sqrt(self._var) + self._epsilon

class ObsNormWrapper(gym.ObservationWrapper):
    """
    Overview:
        The ObsNormWrapper class is a gym observation wrapper that normalizes
        observations according to running mean and standard deviation (std).
    Interfaces:
        __init__, step, reset, observation
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - data_count (:obj:`int`): the count of data points observed so far.
        - clip_range (:obj:`Tuple[int, int]`): the range to clip the normalized observation.
        - rms (:obj:`RunningMeanStd`): running mean and standard deviation of the observations.
    """

    def __init__(self, env: gym.Env):
        """
        Overview:
            Initialize the ObsNormWrapper class.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
        """
        super().__init__(env)
        self.data_count = 0
        self.clip_range = (-3, 3)
        self.rms = RunningMeanStd(shape=env.observation_space.shape)

    def step(self, action: Union[int, np.ndarray]):
        """
        Overview:
            Take an action in the environment, update the running mean and std,
            and return the normalized observation.
        Arguments:
            - action (:obj:`Union[int, np.ndarray]`): the action to take in the environment.
        Returns:
            - obs (:obj:`np.ndarray`): the normalized observation after the action.
            - reward (:obj:`float`): the reward after the action.
            - done (:obj:`bool`): whether the episode has ended.
            - info (:obj:`Dict`): contains auxiliary diagnostic information.
        """
        self.data_count += 1
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.rms.update(obs)
        return self.observation(obs), rew, terminated, truncated, info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Overview:
            Normalize the observation using the current running mean and std.
            If less than 30 data points have been observed, return the original observation.
        Arguments:
            - observation (:obj:`np.ndarray`): the original observation.
        Returns:
            - observation (:obj:`np.ndarray`): the normalized observation.
        """
        if self.data_count > 30:
            return np.clip((observation - self.rms.mean) / self.rms.std, self.clip_range[0], self.clip_range[1])
        else:
            return observation

    def reset(self, **kwargs):
        """
        Overview:
            Reset the environment and the properties related to the running mean and std.
        Arguments:
            - kwargs (:obj:`Dict`): keyword arguments to be passed to the environment's reset function.
        Returns:
            - observation (:obj:`np.ndarray`): the initial observation of the environment.
        """
        # self.data_count = 0
        # self.rms.reset()
        observation = self.env.reset(**kwargs)
        return (self.observation(observation[0]), observation[1])

@ENV_REGISTRY.register('cartpole_lightzero')
class CartPoleEnv(BaseEnv):
    """
    LightZero version of the classic CartPole environment. This class includes methods for resetting, closing, and
    stepping through the environment, as well as seeding for reproducibility, saving replay videos, and generating random
    actions. It also includes properties for accessing the observation space, action space, and reward space of the
    environment.
    """

    config = dict(
        # env_id (str): The name of the environment.
        env_id="CartPole-v0",
        # replay_path (str): The path to save the replay video. If None, the replay will not be saved.
        # Only effective when env_manager.type is 'base'.
        replay_path=None,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict = {}) -> None:
        """
        Initialize the environment with a configuration dictionary. Sets up spaces for observations, actions, and rewards.
        """
        self._cfg = cfg
        self._init_flag = False
        self._continuous = False
        self._replay_path = cfg.replay_path
        self._observation_space = gym.spaces.Box(
            low=np.array([-4.8, float("-inf"), -0.42, float("-inf")]),
            high=np.array([4.8, float("inf"), 0.42, float("inf")]),
            shape=(4,),
            dtype=np.float32
        )
        self._action_space = gym.spaces.Discrete(2)
        self._action_space.seed(0)  # default seed
        self._reward_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment. If it hasn't been initialized yet, this method also handles that. It also handles seeding
        if necessary. Returns the first observation.
        """
        if not self._init_flag:
            self._env = gym.make( self._cfg.env_id, render_mode="rgb_array")
            if self._replay_path is not None:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                video_name = f'{self._env.spec.id}-video-{timestamp}'
                self._env = gym.wrappers.RecordVideo(
                    self._env,
                    video_folder=self._replay_path,
                    episode_trigger=lambda episode_id: True,
                    name_prefix=video_name
                )
            if hasattr(self._cfg, 'obs_plus_prev_action_reward') and self._cfg.obs_plus_prev_action_reward:
                self._env = ObsPlusPrevActRewWrapper(self._env)
            self._init_flag = True
            if hasattr(self._cfg, 'obs_norm') and self._cfg.obs_norm:
                self._env = ObsNormWrapper(self._env)

        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._seed = self._seed + np_seed
            self._action_space.seed(self._seed)
            obs, _ = self._env.reset(seed=self._seed)
        elif hasattr(self, '_seed'):
            self._action_space.seed(self._seed)
            obs, _ = self._env.reset(seed=self._seed)
        else:
            obs, _ = self._env.reset()
        self._observation_space = self._env.observation_space
        self._eval_episode_return = 0
        obs = to_ndarray(obs)

        action_mask = np.ones(self.action_space.n, 'int8')
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        return obs

    def step(self, action: Union[int, np.ndarray]) -> BaseEnvTimestep:
        """
        Overview:
            Perform a step in the environment using the provided action, and return the next state of the environment.
            The next state is encapsulated in a BaseEnvTimestep object, which includes the new observation, reward,
            done flag, and info dictionary.
        Arguments:
            - action (:obj:`Union[int, np.ndarray]`): The action to be performed in the environment. If the action is
              a 1-dimensional numpy array, it is squeezed to a 0-dimension array.
        Returns:
            - timestep (:obj:`BaseEnvTimestep`): An object containing the new observation, reward, done flag,
              and info dictionary.
        .. note::
            - The cumulative reward (`_eval_episode_return`) is updated with the reward obtained in this step.
            - If the episode ends (done is True), the total reward for the episode is stored in the info dictionary
              under the key 'eval_episode_return'.
            - An action mask is created with ones, which represents the availability of each action in the action space.
            - Observations are returned in a dictionary format containing 'observation', 'action_mask', and 'to_play'.
        """
        if isinstance(action, np.ndarray) and action.shape == (1,):
            action = action.squeeze()  # 0-dim array

        obs, rew, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated

        self._eval_episode_return += rew
        if done:
            info['eval_episode_return'] = self._eval_episode_return

        action_mask = np.ones(self.action_space.n, 'int8')
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        return BaseEnvTimestep(obs, rew, done, info)

    def close(self) -> None:
        """
        Close the environment, and set the initialization flag to False.
        """
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Set the seed for the environment's random number generator. Can handle both static and dynamic seeding.
        """
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        """
        Enable the saving of replay videos. If no replay path is given, a default is used.
        """
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    def random_action(self) -> np.ndarray:
        """
         Generate a random action using the action space's sample method. Returns a numpy array containing the action.
         """
        random_action = self.action_space.sample()
        random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    @property
    def observation_space(self) -> gym.spaces.Space:
        """
        Property to access the observation space of the environment.
        """
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        """
        Property to access the action space of the environment.
        """
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        """
        Property to access the reward space of the environment.
        """
        return self._reward_space

    def __repr__(self) -> str:
        """
        String representation of the environment.
        """
        return "LightZero CartPole Env"
