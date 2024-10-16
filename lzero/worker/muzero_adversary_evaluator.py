import copy
import time
from collections import namedtuple
from typing import Optional, Callable, Tuple, Dict, Any

import numpy as np
import torch
from ding.envs import BaseEnvManager
from ding.torch_utils import to_ndarray, to_item, to_tensor
from ding.utils import build_logger, EasyTimer
from ding.utils import get_world_size, get_rank, broadcast_object_list
from ding.worker.collector.base_serial_evaluator import ISerialEvaluator, VectorEvalMonitor
from easydict import EasyDict

from ATLA_robust_RL.src import apply_ppo_adversary
from lzero.mcts.buffer.adversary_game_segment import AdversaryGameSegment as GameSegment
from lzero.mcts.utils import prepare_observation


class MuZeroAdversaryEvaluator(ISerialEvaluator):
    """
    Overview:
        The Evaluator class for MCTS+RL algorithms, such as MuZero, EfficientZero, and Sampled EfficientZero.
    Interfaces:
        __init__, reset, reset_policy, reset_env, close, should_eval, eval
    Properties:
        env, policy
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        """
        Overview:
            Retrieve the default configuration for the evaluator by merging evaluator-specific defaults with other
            defaults and any user-provided configuration.
        Returns:
            - cfg (:obj:`EasyDict`): The default configuration for the evaluator.
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        # Evaluate every "eval_freq" training iterations.
        eval_freq=50,
    )

    def __init__(
            self,
            eval_freq: int = 1000,
            n_evaluator_episode: int = 3,
            stop_value: int = 1e6,
            env: BaseEnvManager = None,
            policy: namedtuple = None,
            policy_adversary: namedtuple = None,
            tb_logger: 'SummaryWriter' = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'evaluator',
            policy_config: 'policy_config' = None,  # noqa
            policy_adversary_config: 'policy_config' = None,  # noqa
    ) -> None:
        """
        Overview:
            Initialize the evaluator with configuration settings for various components such as logger helper and timer.
        Arguments:
            - eval_freq (:obj:`int`): Evaluation frequency in terms of training steps.
            - n_evaluator_episode (:obj:`int`): Number of episodes to evaluate in total.
            - stop_value (:obj:`float`): A reward threshold above which the training is considered converged.
            - env (:obj:`Optional[BaseEnvManager]`): An optional instance of a subclass of BaseEnvManager.
            - policy (:obj:`Optional[namedtuple]`): An optional API namedtuple defining the policy for evaluation.
            - tb_logger (:obj:`Optional[SummaryWriter]`): Optional TensorBoard logger instance.
            - exp_name (:obj:`str`): Name of the experiment, used to determine output directory.
            - instance_name (:obj:`str`): Name of this evaluator instance.
            - policy_config (:obj:`Optional[dict]`): Optional configuration for the game policy.
        """
        self._eval_freq = eval_freq
        self._exp_name = exp_name
        self._instance_name = instance_name

        # Logger (Monitor will be initialized in policy setter)
        # Only rank == 0 learner needs monitor and tb_logger, others only need text_logger to display terminal output.
        if get_rank() == 0:
            if tb_logger is not None:
                self._logger, _ = build_logger(
                    './{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name, need_tb=False
                )
                self._tb_logger = tb_logger
            else:
                self._logger, self._tb_logger = build_logger(
                    './{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name
                )
        else:
            self._logger, self._tb_logger = None, None  # for close elegantly

        self.reset(policy, policy_adversary, env)

        self._timer = EasyTimer()
        self._default_n_episode = n_evaluator_episode
        self._stop_value = stop_value

        # ==============================================================
        # MCTS+RL related core code
        # ==============================================================
        self.policy_config = policy_config
        if policy_adversary_config is not None:
            self._epsilon = policy_adversary_config.Epsilon
            self._noise_policy = policy_adversary_config.noise_policy

            if self._noise_policy == 'ppo':
                self.pretrained_ppo_adversary = apply_ppo_adversary.main(policy_adversary_config.env_seed,policy_adversary_config.ppo_adv_config_path,
                                                                    policy_adversary_config.attack_method, policy_adversary_config.attack_advpolicy_network,
                                                                    policy_adversary_config.action_shape, policy_adversary_config.obs_shape, policy_adversary_config.action_space)

    def reset_env(self, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset the environment for the evaluator, optionally replacing it with a new environment.
            If _env is None, reset the old environment. If _env is not None, replace the old environment
            in the evaluator with the new passed in environment and launch.
        Arguments:
            - _env (:obj:`Optional[BaseEnvManager]`): An optional new environment instance to replace the existing one.
        """
        if _env is not None:
            self._env = _env
            self._env.launch()
            self._env_num = self._env.env_num
        else:
            self._env.reset()

    def reset_policy(self, _policy: Optional[namedtuple] = None, _policy_adversary: Optional[namedtuple] = None) -> None:
        """
        Overview:
            Reset the policy for the evaluator, optionally replacing it with a new policy.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the evaluator with the new passed in policy.
        Arguments:
            - _policy (:obj:`Optional[namedtuple]`): An optional new policy namedtuple to replace the existing one.
        """
        assert hasattr(self, '_env'), "please set env first"
        if _policy is not None:
            self._policy = _policy
        if _policy_adversary is not None:
            self._policy_adversary = _policy_adversary
            self._policy_adversary.reset()
        self._policy.reset()

    def reset(self, _policy: Optional[namedtuple] = None, _policy_adversary: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset both the policy and environment for the evaluator, optionally replacing them.
            If _env is None, reset the old environment.
            If _env is not None, replace the old environment in the evaluator with the new passed in \
                environment and launch.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the evaluator with the new passed in policy.
        Arguments:
            - _policy (:obj:`Optional[namedtuple]`): An optional new policy namedtuple to replace the existing one.
            - _env (:obj:`Optional[BaseEnvManager]`): An optional new environment instance to replace the existing one.
        """
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy, _policy_adversary)
        self._max_episode_return = float("-inf")
        self._last_eval_iter = 0
        self._end_flag = False

    def close(self) -> None:
        """
        Overview:
            Close the evaluator, the environment, flush and close the TensorBoard logger if applicable.
        """
        if self._end_flag:
            return
        self._end_flag = True
        self._env.close()
        if self._tb_logger:
            self._tb_logger.flush()
            self._tb_logger.close()

    def __del__(self):
        """
        Overview:
            Execute the close command and close the evaluator. __del__ is automatically called \
                to destroy the evaluator instance when the evaluator finishes its work
        """
        self.close()

    def should_eval(self, train_iter: int) -> bool:
        """
        Overview:
            Determine whether to initiate evaluation based on the training iteration count and evaluation frequency.
        Arguments:
            - train_iter (:obj:`int`): The current count of training iterations.
        Returns:
            - (:obj:`bool`): `True` if evaluation should be initiated, otherwise `False`.
        """
        if train_iter == self._last_eval_iter:
            return False
        if (train_iter - self._last_eval_iter) < self._eval_freq and train_iter != 0:
            return False
        self._last_eval_iter = train_iter
        return True

    def eval(
            self,
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Overview:
            Evaluate the current policy, storing the best policy if it achieves the highest historical reward.
        Arguments:
            - save_ckpt_fn (:obj:`Optional[Callable]`): Optional function to save a checkpoint when a new best reward is achieved.
            - train_iter (:obj:`int`): The current training iteration count.
            - envstep (:obj:`int`): The current environment step count.
            - n_episode (:obj:`Optional[int]`): Optional number of evaluation episodes; defaults to the evaluator's setting.
        Returns:
            - stop_flag (:obj:`bool`): Indicates whether the training can be stopped based on the stop value.
            - episode_info (:obj:`Dict[str, Any]`): A dictionary containing information about the evaluation episodes.
        """
        # the evaluator only works on rank0
        episode_info = None
        stop_flag = False
        if get_rank() == 0:
            if n_episode is None:
                n_episode = self._default_n_episode
            assert n_episode is not None, "please indicate eval n_episode"
            envstep_count = 0
            eval_monitor = VectorEvalMonitor(self._env.env_num, n_episode)
            env_nums = self._env.env_num

            self._env.reset()
            self._policy.reset()

            # initializations
            init_obs = self._env.ready_obs

            retry_waiting_time = 0.001
            while len(init_obs.keys()) != self._env_num:
                # To be compatible with subprocess env_manager, in which sometimes self._env_num is not equal to
                # len(self._env.ready_obs), especially in tictactoe env.
                self._logger.info('The current init_obs.keys() is {}'.format(init_obs.keys()))
                self._logger.info('Before sleeping, the _env_states is {}'.format(self._env._env_states))
                time.sleep(retry_waiting_time)
                self._logger.info('=' * 10 + 'Wait for all environments (subprocess) to finish resetting.' + '=' * 10)
                self._logger.info(
                    'After sleeping {}s, the current _env_states is {}'.format(retry_waiting_time,
                                                                               self._env._env_states)
                )
                init_obs = self._env.ready_obs

            action_mask_dict = {i: to_ndarray(init_obs[i]['action_mask']) for i in range(env_nums)}

            to_play_dict = {i: to_ndarray(init_obs[i]['to_play']) for i in range(env_nums)}
            dones = np.array([False for _ in range(env_nums)])

            game_segments = [
                GameSegment(
                    self._env.action_space,
                    game_segment_length=self.policy_config.game_segment_length,
                    config=self.policy_config
                ) for _ in range(env_nums)
            ]
            for i in range(env_nums):
                game_segments[i].reset(
                    [to_ndarray(init_obs[i]['observation']) for _ in range(self.policy_config.model.frame_stack_num)]
                )

                # ============================================
                # first add Adverse
                # ====================================================
                if hasattr(self, '_noise_policy'):
                    if self._noise_policy == 'atla_ppo':
                        stack_obs = {env_id: init_obs[env_id]['observation'] for env_id in range(env_nums)}
                        noise = self._policy_adversary.forward(stack_obs)
                        for env_id in noise.keys():
                            noise[env_id]['action'] = torch.clamp(noise[env_id]['action'], -1 * self._epsilon,
                                                                  self._epsilon)
                            game_segments[env_id].reset([to_ndarray(init_obs[i]['observation'] + noise[env_id]['action'].tolist())
                                                         for _ in range(self.policy_config.model.frame_stack_num)],
                                                        [to_ndarray(init_obs[i]['observation']) for _ in
                                                         range(self.policy_config.model.frame_stack_num)])
                    elif self._noise_policy == 'ppo':
                        stack_obs = {env_id: init_obs[env_id]['observation'] for env_id in range(env_nums)}
                        stacked_tensor = torch.stack([torch.tensor(stack_obs[key]) for key in range(env_nums)]).float().to(self.policy_config.device)
                        perturbations_mean = self.pretrained_ppo_adversary.apply_ppo_attack(stacked_tensor)
                        for env_id in range(env_nums):
                            # Clamp using tanh.
                            noise = torch.nn.functional.hardtanh(perturbations_mean[env_id]) * self._epsilon
                            noise = torch.clamp(noise, -1 * self._epsilon, self._epsilon)
                            game_segments[env_id].reset(
                                [to_ndarray(init_obs[i]['observation'] + noise.cpu().tolist())
                                 for _ in range(self.policy_config.model.frame_stack_num)],
                                [to_ndarray(init_obs[i]['observation']) for _ in
                                 range(self.policy_config.model.frame_stack_num)])
                    elif self._noise_policy == 'random':
                        for env_id in range(env_nums):
                            # noise = np.random.normal(0, 0.001, len(init_obs[env_id]['observation']))
                            # noise = torch.clamp(torch.Tensor(noise), -1 * self._epsilon, self._epsilon).cpu()
                            noise = np.random.uniform(-1 * self._epsilon, self._epsilon, len(init_obs[env_id]['observation']))
                            noise = torch.Tensor(noise).cpu()
                            game_segments[env_id].reset(
                                [to_ndarray(init_obs[i]['observation'] + noise.numpy())
                                 for _ in range(self.policy_config.model.frame_stack_num)],
                                [to_ndarray(init_obs[i]['observation']) for _ in
                                 range(self.policy_config.model.frame_stack_num)])

            ready_env_id = set()
            remain_episode = n_episode

            env_step = 0
            with self._timer:
                while not eval_monitor.is_finished():
                    # Get current ready env obs.
                    if (env_step % 50 == 0):
                        print(env_step)
                    env_step += 1
                    obs = self._env.ready_obs
                    new_available_env_id = set(obs.keys()).difference(ready_env_id)
                    ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                    remain_episode -= min(len(new_available_env_id), remain_episode)

                    stack_obs = {env_id: game_segments[env_id].get_obs() for env_id in ready_env_id}
                    stack_obs = list(stack_obs.values())

                    action_mask_dict = {env_id: action_mask_dict[env_id] for env_id in ready_env_id}
                    to_play_dict = {env_id: to_play_dict[env_id] for env_id in ready_env_id}
                    action_mask = [action_mask_dict[env_id] for env_id in ready_env_id]
                    to_play = [to_play_dict[env_id] for env_id in ready_env_id]

                    stack_obs = to_ndarray(stack_obs)
                    stack_obs = prepare_observation(stack_obs, self.policy_config.model.model_type)
                    stack_obs = torch.from_numpy(stack_obs).to(self.policy_config.device).float()

                    # ==============================================================
                    # policy forward
                    # ==============================================================
                    policy_output = self._policy.forward(stack_obs, action_mask, to_play)

                    actions_no_env_id = {k: v['action'] for k, v in policy_output.items()}
                    distributions_dict_no_env_id = {k: v['visit_count_distributions'] for k, v in policy_output.items()}
                    if self.policy_config.sampled_algo:
                        root_sampled_actions_dict_no_env_id = {
                            k: v['root_sampled_actions']
                            for k, v in policy_output.items()
                        }

                    value_dict_no_env_id = {k: v['searched_value'] for k, v in policy_output.items()}
                    pred_value_dict_no_env_id = {k: v['predicted_value'] for k, v in policy_output.items()}
                    visit_entropy_dict_no_env_id = {
                        k: v['visit_count_distribution_entropy']
                        for k, v in policy_output.items()
                    }

                    actions = {}
                    distributions_dict = {}
                    if self.policy_config.sampled_algo:
                        root_sampled_actions_dict = {}
                    value_dict = {}
                    pred_value_dict = {}
                    visit_entropy_dict = {}
                    for index, env_id in enumerate(ready_env_id):
                        actions[env_id] = actions_no_env_id.pop(index)
                        distributions_dict[env_id] = distributions_dict_no_env_id.pop(index)
                        if self.policy_config.sampled_algo:
                            root_sampled_actions_dict[env_id] = root_sampled_actions_dict_no_env_id.pop(index)
                        value_dict[env_id] = value_dict_no_env_id.pop(index)
                        pred_value_dict[env_id] = pred_value_dict_no_env_id.pop(index)
                        visit_entropy_dict[env_id] = visit_entropy_dict_no_env_id.pop(index)

                    # ==============================================================
                    # Interact with env.
                    # ==============================================================
                    timesteps = self._env.step(actions)
                    timesteps = to_tensor(timesteps, dtype=torch.float32)
                    # TODO
                    # ==============================================================
                    # Adding Noise for observation by utilizing PPO Adversary.
                    # ==============================================================
                    if hasattr(self, '_noise_policy'):
                        if self._noise_policy == 'atla_ppo':
                            timesteps_copy = copy.deepcopy(timesteps)
                            for env_id in timesteps_copy.keys():
                                timesteps_copy[env_id] = timesteps_copy[env_id].obs['observation']
                            noise = self._policy_adversary.forward(timesteps_copy)
                            for env_id in noise.keys():
                                noise[env_id]['action'] = torch.clamp(noise[env_id]['action'], -1 * self._epsilon,
                                                                      self._epsilon)
                                timesteps[env_id].obs['observation_true'] = timesteps[env_id].obs['observation']
                                timesteps[env_id].obs['observation'] = timesteps[env_id].obs['observation'] + \
                                                                       noise[env_id]['action'].numpy()
                        elif self._noise_policy == 'ppo':
                            timesteps_copy = copy.deepcopy(timesteps)
                            for env_id in timesteps_copy.keys():
                                timesteps_copy[env_id] = timesteps_copy[env_id].obs['observation']
                            stacked_tensor = torch.stack(
                                [timesteps_copy[key].detach().clone() for key in timesteps_copy.keys()]).to(self.policy_config.device)
                            perturbations_mean = self.pretrained_ppo_adversary.apply_ppo_attack(stacked_tensor)
                            for idx, env_id in enumerate(timesteps.keys()):
                                # Clamp using tanh.
                                noise = torch.nn.functional.hardtanh(perturbations_mean[idx]) * self._epsilon
                                noise = torch.clamp(noise, -1 * self._epsilon, self._epsilon)
                                timesteps[env_id].obs['observation_true'] = timesteps[env_id].obs['observation']
                                timesteps[env_id].obs['observation'] = timesteps[env_id].obs['observation'] + noise.detach().cpu()
                        elif self._noise_policy == 'random':
                            for env_id in timesteps.keys():
                                # noise = np.random.normal(0, self._epsilon/3.0, len(timesteps[env_id].obs['observation']))
                                # noise = torch.clamp(torch.Tensor(noise), -1 * self._epsilon, self._epsilon).cpu()
                                noise = np.random.uniform(-1 * self._epsilon, self._epsilon, len(init_obs[env_id]['observation']))
                                noise = torch.Tensor(noise).cpu()
                                timesteps[env_id].obs['observation_true'] = timesteps[env_id].obs['observation']
                                timesteps[env_id].obs['observation'] = timesteps[env_id].obs[
                                                                           'observation'] + noise.numpy()

                    for env_id, t in timesteps.items():
                        obs, reward, done, info = t.obs, t.reward, t.done, t.info

                        game_segments[env_id].append(
                            actions[env_id], to_ndarray(obs['observation']), reward, action_mask_dict[env_id],
                            to_play_dict[env_id],  true_obs=to_ndarray(obs['observation_true']) if hasattr(self, '_noise_policy') else None
                        )

                        # NOTE: in evaluator, we only need save the ``o_{t+1} = obs['observation']``
                        # game_segments[env_id].obs_segment.append(to_ndarray(obs['observation']))

                        # NOTE: the position of code snippet is very important.
                        # the obs['action_mask'] and obs['to_play'] are corresponding to next action
                        action_mask_dict[env_id] = to_ndarray(obs['action_mask'])
                        to_play_dict[env_id] = to_ndarray(obs['to_play'].cpu())

                        dones[env_id] = done
                        if t.done:
                            # Env reset is done by env_manager automatically.
                            self._policy.reset([env_id])
                            reward = t.info['eval_episode_return']
                            saved_info = {'eval_episode_return': t.info['eval_episode_return']}
                            if 'episode_info' in t.info:
                                saved_info.update(t.info['episode_info'])
                            eval_monitor.update_info(env_id, saved_info)
                            eval_monitor.update_reward(env_id, reward)
                            self._logger.info(
                                "[EVALUATOR]env {} finish episode, final reward: {}, current episode: {}".format(
                                    env_id, eval_monitor.get_latest_reward(env_id), eval_monitor.get_current_episode()
                                )
                            )

                            # reset the finished env and init game_segments
                            if n_episode > self._env_num:
                                # Get current ready env obs.
                                init_obs = self._env.ready_obs
                                retry_waiting_time = 0.001
                                while len(init_obs.keys()) != self._env_num:
                                    # In order to be compatible with subprocess env_manager, in which sometimes self._env_num is not equal to
                                    # len(self._env.ready_obs), especially in tictactoe env.
                                    self._logger.info('The current init_obs.keys() is {}'.format(init_obs.keys()))
                                    self._logger.info(
                                        'Before sleeping, the _env_states is {}'.format(self._env._env_states)
                                    )
                                    time.sleep(retry_waiting_time)
                                    self._logger.info(
                                        '=' * 10 + 'Wait for all environments (subprocess) to finish resetting.' + '=' * 10
                                    )
                                    self._logger.info(
                                        'After sleeping {}s, the current _env_states is {}'.format(
                                            retry_waiting_time, self._env._env_states
                                        )
                                    )
                                    init_obs = self._env.ready_obs

                                new_available_env_id = set(init_obs.keys()).difference(ready_env_id)
                                ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                                remain_episode -= min(len(new_available_env_id), remain_episode)

                                action_mask_dict[env_id] = to_ndarray(init_obs[env_id]['action_mask'])
                                to_play_dict[env_id] = to_ndarray(init_obs[env_id]['to_play'])

                                game_segments[env_id] = GameSegment(
                                    self._env.action_space,
                                    game_segment_length=self.policy_config.game_segment_length,
                                    config=self.policy_config
                                )
                                if hasattr(self, '_noise_policy'):
                                    if self._noise_policy == 'atla_ppo':
                                        stack_obs = {env_id: init_obs[env_id]['observation']}
                                        noise = self._policy_adversary.forward(stack_obs)
                                        noise[env_id]['action'] = torch.clamp(noise[env_id]['action'], -1 * self._epsilon, self._epsilon)
                                        game_segments[env_id].reset(
                                            [to_ndarray(init_obs[i]['observation'] + noise[env_id]['action'].tolist())
                                             for _ in range(self.policy_config.model.frame_stack_num)],
                                            [to_ndarray(init_obs[i]['observation']) for _ in
                                             range(self.policy_config.model.frame_stack_num)])

                                    elif self._noise_policy == 'ppo':
                                        stack_obs = {env_id: init_obs[env_id]['observation']}
                                        stacked_tensor = torch.stack(
                                            [torch.tensor(stack_obs[key]) for key in stack_obs.keys()]).to(self.policy_config.device)
                                        perturbations_mean = self.pretrained_ppo_adversary.apply_ppo_attack(
                                            stacked_tensor)
                                        noise = torch.nn.functional.hardtanh(perturbations_mean[0]) * self._epsilon
                                        noise = torch.clamp(noise, -1 * self._epsilon, self._epsilon)
                                        game_segments[env_id].reset(
                                            [to_ndarray(init_obs[i]['observation'] + noise.cpu().tolist())
                                             for _ in range(self.policy_config.model.frame_stack_num)],
                                            [to_ndarray(init_obs[i]['observation']) for _ in
                                             range(self.policy_config.model.frame_stack_num)])
                                    elif self._noise_policy == 'random':
                                        # noise = np.random.normal(0, 0.001, len(init_obs[env_id]['observation']))
                                        # noise = torch.clamp(torch.Tensor(noise), -1 * self._epsilon, self._epsilon).cpu()
                                        noise = np.random.uniform(-1 * self._epsilon, self._epsilon, len(init_obs[env_id]['observation']))
                                        noise = torch.Tensor(noise).cpu()
                                        game_segments[env_id].reset(
                                            [to_ndarray(init_obs[i]['observation'] + noise.numpy())
                                             for _ in range(self.policy_config.model.frame_stack_num)],
                                            [to_ndarray(init_obs[i]['observation']) for _ in
                                             range(self.policy_config.model.frame_stack_num)])

                                else:
                                    game_segments[env_id].reset(
                                        [
                                            init_obs[env_id]['observation']
                                            for _ in range(self.policy_config.model.frame_stack_num)
                                        ]
                                    )

                            # Env reset is done by env_manager automatically.
                            self._policy.reset([env_id])
                            # TODO(pu): subprocess mode, when n_episode > self._env_num, occasionally the ready_env_id=()
                            # and the stack_obs is np.array(None, dtype=object)
                            ready_env_id.remove(env_id)

                        envstep_count += 1
            duration = self._timer.value
            episode_return = eval_monitor.get_episode_return()
            info = {
                'train_iter': train_iter,
                'ckpt_name': 'iteration_{}.pth.tar'.format(train_iter),
                'episode_count': n_episode,
                'envstep_count': envstep_count,
                'avg_envstep_per_episode': envstep_count / n_episode,
                'evaluate_time': duration,
                'avg_envstep_per_sec': envstep_count / duration,
                'avg_time_per_episode': n_episode / duration,
                'reward_mean': np.mean(episode_return),
                'reward_std': np.std(episode_return),
                'reward_max': np.max(episode_return),
                'reward_min': np.min(episode_return),
                # 'each_reward': episode_return,
            }
            episode_info = eval_monitor.get_episode_info()
            if episode_info is not None:
                info.update(episode_info)
            self._logger.info(self._logger.get_tabulate_vars_hor(info))
            # self._logger.info(self._logger.get_tabulate_vars(info))
            for k, v in info.items():
                if k in ['train_iter', 'ckpt_name', 'each_reward']:
                    continue
                if not np.isscalar(v):
                    continue
                self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
                self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, envstep)
            episode_return = np.mean(episode_return)
            if episode_return > self._max_episode_return:
                if save_ckpt_fn:
                    save_ckpt_fn(f'ckpt_best_{self._instance_name}.pth.tar')
                self._max_episode_return = episode_return
            stop_flag = episode_return >= self._stop_value and train_iter > 0
            if stop_flag:
                self._logger.info(
                    "[LightZero serial pipeline] " +
                    "Current episode_return: {} is greater than stop_value: {}".format(episode_return,
                                                                                       self._stop_value) +
                    ", so your MCTS/RL agent is converged, you can refer to 'log/evaluator/evaluator_logger.txt' for details."
                )

        if get_world_size() > 1:
            objects = [stop_flag, episode_info]
            broadcast_object_list(objects, src=0)
            stop_flag, episode_info = objects

        episode_info = to_item(episode_info)
        return stop_flag, episode_info
