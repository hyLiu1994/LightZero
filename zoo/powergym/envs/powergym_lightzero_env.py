import copy
import os
import random
from typing import Union, List, Dict, Any, Optional

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnvTimestep, BaseEnv
from ding.envs.common import save_frames_as_gif
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from matplotlib import pyplot as plt

from zoo.powergym.envs.powergym.env_register import make_env, get_info_and_folder


@ENV_REGISTRY.register('powergym_lightzero')
class PowerGymEnvLZ(BaseEnv):
    """
    Overview:
        The modified powerGym environment with continuous action space for LightZero's algorithms.
    """
    config = dict(
        stop_value=int(1e6),
        action_clip=False,
        action_bins_per_branch=None,

        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),

        # env_name='13Bus',
        # do_testing=False,
        use_plot=False,

    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            Initialize the MuJoCo environment.
        Arguments:
            - cfg (:obj:`dict`): Configuration dict. The dict should include keys like 'env_id', 'replay_path', etc.
        """
        self._cfg = cfg
        # We use env_id to indicate the env_id in LightZero.
        self._env_id = self._cfg.env_id

        self._action_clip = cfg.action_clip
        self._action_bins_per_branch = cfg.action_bins_per_branch

        self._init_flag = False

        # self._do_testing = cfg.do_testing
        self._use_plot = cfg.use_plot

        #   self._env_ref = self._env_fn[0]() 只有进行异步初始化时才能进行相关cfg.work_idx
        if hasattr(cfg, 'work_idx'): # 这个同步操作时会初始化work_idx = 0 , 后面环境则直接使用.
            self._worker_idx = cfg.work_idx
        else:
            self._worker_idx = 0
        self._save_replay_count = 0


    def reset(self) -> Dict[str, Union[Optional[int], Any]]:
        """
        Overview:
            Reset the environment and return the initial observation.
        Returns:
            - obs (:obj:`np.ndarray`): The initial observation after resetting.
        """
        if not self._init_flag:
            self._env = make_env(self._env_id, dss_act=False, worker_idx=self._worker_idx)
            self._env.observation_space.dtype = np.float32
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1,), dtype=np.float32
            )
            self._init_flag = True

            # if self._do_testing:
            #     self.train_profiles = random.sample(range(self._env.num_profiles), k=self._env.num_profiles // 2)
            #     self.test_profiles = [i for i in range(self._env.num_profiles) if i not in self.train_profiles]
            # else:
            #     self.train_profiles = list(range(self._env.num_profiles))

        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)

        # load_profile_idx = random.choice(train_profiles)
        obs = self._env.reset(load_profile_idx = self._worker_idx)
        obs = to_ndarray(obs).astype('float32')
        self._eval_episode_return = 0.

        action_mask = None
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        return obs

    def step(self, action: Union[np.ndarray, list]) -> BaseEnvTimestep:
        """
        Overview:
            Perform a step in the environment using the provided action, and return the next state of the environment.
            The next state is encapsulated in a BaseEnvTimestep object, which includes the new observation, reward,
            done flag, and info dictionary.
        Arguments:
            - action (:obj:`Union[np.ndarray, list]`): The action to be performed in the environment. 
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
        # if self._action_bins_per_branch:
        #     action = self.map_action(action)
        '''
         action [-0.63713658 -0.98315281  0.65262783  0.04777701 -0.16687974 -0.81025612]
         action [ 0  0 26 17 13  3]
        '''
        # print("action", action)
        # action 出来的是 [-1,1] 的值 如何映射到离散值0,1 以及0到32
        action = self.map_action(action)
        action = to_ndarray(action)
        # print("action", action)

        # if self._action_clip:
        #     action = np.clip(action, -1, 1)

        obs, rew, done, info = self._env.step(action)
        self._eval_episode_return += rew
        if done:
            if self._use_plot:
                path = os.path.join(
                    os.path.join(os.getcwd(), 'agent_plots'), '{}_{}_episode_{}.png'
                    .format(self._cfg.env_id, str(self._cfg.env_id).zfill(4),self._save_replay_count)
                )
                fig, _ = self._env.plot_graph()
                fig.tight_layout(pad=0.1)
                fig.savefig(path)
                plt.close()

                self._save_replay_count += 1
            info['eval_episode_return'] = self._eval_episode_return

        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([rew]).astype(np.float32)

        action_mask = None
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        return BaseEnvTimestep(obs, rew, done, info)

    def random_action(self) -> np.ndarray:
        return self._env.random_action()

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_cfg = copy.deepcopy(cfg)
        collector_env_num = collector_cfg.pop('collector_env_num', 1)
        return [collector_cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_cfg = copy.deepcopy(cfg)
        evaluator_env_num = evaluator_cfg.pop('evaluator_env_num', 1)
        evaluator_cfg.norm_reward.use_norm = False
        return [evaluator_cfg for _ in range(evaluator_env_num)]

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        """
        String representation of the environment.
        """
        return "LightZero Mujoco Env({})".format(self._cfg.env_id)

    def map_action(self, action): # [map_to_discrete(val) for val in test_values]
        if self._env_id == '13Bus' or self._env_id == '34Bus':
            # 取出 ndarray 取出前面两个动作, 映射到0, 1 后面四个动作映射到0,32
            two_action_list = [map_to_discrete(val, 2) for val in action[0:2]]
            four_action_list = [map_to_discrete(val, 32) for val in action[2:]]
        elif self._env_id == '123Bus':
            two_action_list = [map_to_discrete(val, 2) for val in action[0:4]]
            four_action_list = [map_to_discrete(val, 32) for val in action[4:]]
        elif self._env_id == '8500Node':
            two_action_list = [map_to_discrete(val, 2) for val in action[0:10]]
            four_action_list = [map_to_discrete(val, 32) for val in action[10:]]

        return two_action_list + four_action_list



def map_to_discrete(value, threshold):
    # 第一步：归一化，将 [-1, 1] 映射到 [0, 1]
    normalized = (value + 1) / 2
    # 第二步：缩放，将 [0, 1] 映射到 离散值 {0,1} 或者{0, 32}
    scaled = normalized * threshold
    # 将结果转换为整数
    discrete_value = int(round(scaled))
    # 确保结果在 0 到 32 的范围内
    discrete_value = min(max(discrete_value, 0), threshold)

    return discrete_value