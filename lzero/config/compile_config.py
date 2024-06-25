import os
import datetime
from easydict import EasyDict
from copy import deepcopy

from ding.utils import deep_merge_dicts, get_rank
from ding.envs import get_env_cls, get_env_manager_cls, BaseEnvManager
from ding.policy import get_policy_cls
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, Coordinator, \
    AdvancedReplayBuffer, get_parallel_commander_cls, get_parallel_collector_cls, get_buffer_cls, \
    get_serial_collector_cls, MetricSerialEvaluator, BattleInteractionSerialEvaluator
from ding.reward_model import get_reward_model_cls
from ding.world_model import get_world_model_cls
from ding.config.config import compile_buffer_config, policy_config_template, \
    env_config_template, compile_collector_config, save_project_state, save_config

def compile_config(
        cfg: EasyDict,
        env_manager: type = None,
        policy: type = None,
        policy_adversary: type = None,
        learner: type = BaseLearner,
        collector: type = None,
        evaluator: type = InteractionSerialEvaluator,
        buffer: type = None,
        env: type = None,
        reward_model: type = None,
        world_model: type = None,
        seed: int = 0,
        auto: bool = False,
        create_cfg: dict = None,
        save_cfg: bool = True,
        save_path: str = 'total_config.py',
        renew_dir: bool = True,
        have_adversary: bool = False,
) -> EasyDict:
    """
    Overview:
        Combine the input config information with other input information.
        Compile config to make it easy to be called by other programs
    Arguments:
        - cfg (:obj:`EasyDict`): Input config dict which is to be used in the following pipeline
        - env_manager (:obj:`type`): Env_manager class which is to be used in the following pipeline
        - policy (:obj:`type`): Policy class which is to be used in the following pipeline
        - learner (:obj:`type`): Input learner class, defaults to BaseLearner
        - collector (:obj:`type`): Input collector class, defaults to BaseSerialCollector
        - evaluator (:obj:`type`): Input evaluator class, defaults to InteractionSerialEvaluator
        - buffer (:obj:`type`): Input buffer class, defaults to IBuffer
        - env (:obj:`type`): Environment class which is to be used in the following pipeline
        - reward_model (:obj:`type`): Reward model class which aims to offer various and valuable reward
        - seed (:obj:`int`): Random number seed
        - auto (:obj:`bool`): Compile create_config dict or not
        - create_cfg (:obj:`dict`): Input create config dict
        - save_cfg (:obj:`bool`): Save config or not
        - save_path (:obj:`str`): Path of saving file
        - renew_dir (:obj:`bool`): Whether to new a directory for saving config.
    Returns:
        - cfg (:obj:`EasyDict`): Config after compiling
    """
    cfg, create_cfg = deepcopy(cfg), deepcopy(create_cfg)
    if auto:
        assert create_cfg is not None
        # for compatibility
        if 'collector' not in create_cfg:
            create_cfg.collector = EasyDict(dict(type='sample'))
        if 'replay_buffer' not in create_cfg:
            create_cfg.replay_buffer = EasyDict(dict(type='advanced'))
            buffer = AdvancedReplayBuffer
        if env is None:
            if 'env' in create_cfg:
                env = get_env_cls(create_cfg.env)
            else:
                env = None
                create_cfg.env = {'type': 'ding_env_wrapper_generated'}
        if env_manager is None:
            env_manager = get_env_manager_cls(create_cfg.env_manager)
            print("env_manager", env_manager)
        if policy is None:
            policy = get_policy_cls(create_cfg.policy)
            if have_adversary:
                policy_adversary = get_policy_cls(create_cfg.policy_adversary)
        if 'default_config' in dir(env):
            env_config = env.default_config()
        else:
            env_config = EasyDict()  # env does not have default_config
        env_config = deep_merge_dicts(env_config_template, env_config)
        env_config.update(create_cfg.env)
        env_config.manager = deep_merge_dicts(env_manager.default_config(), env_config.manager)
        env_config.manager.update(create_cfg.env_manager)
        policy_config = policy.default_config()
        policy_config = deep_merge_dicts(policy_config_template, policy_config)
        policy_config.update(create_cfg.policy)
        policy_config.collect.collector.update(create_cfg.collector)
        if 'evaluator' in create_cfg:
            policy_config.eval.evaluator.update(create_cfg.evaluator)
        policy_config.other.replay_buffer.update(create_cfg.replay_buffer)

        policy_config.other.commander = BaseSerialCommander.default_config()

        if have_adversary:
            policy_adversary_config = policy_adversary.default_config()
            policy_adversary_config = deep_merge_dicts(policy_config_template, policy_adversary_config)
            policy_adversary_config.update(create_cfg.policy_adversary)
            policy_adversary_config.collect.collector.update(create_cfg.collector)
            if 'evaluator' in create_cfg:
                policy_adversary_config.eval.evaluator.update(create_cfg.evaluator)
            policy_adversary_config.other.replay_buffer.update(create_cfg.replay_buffer)

            policy_adversary_config.other.commander = BaseSerialCommander.default_config()

        if 'reward_model' in create_cfg:
            reward_model = get_reward_model_cls(create_cfg.reward_model)
            reward_model_config = reward_model.default_config()
        else:
            reward_model_config = EasyDict()
        if 'world_model' in create_cfg:
            world_model = get_world_model_cls(create_cfg.world_model)
            world_model_config = world_model.default_config()
            world_model_config.update(create_cfg.world_model)
        else:
            world_model_config = EasyDict()
    else:
        if 'default_config' in dir(env):
            env_config = env.default_config()
        else:
            env_config = EasyDict()  # env does not have default_config
        env_config = deep_merge_dicts(env_config_template, env_config)
        if env_manager is None:
            env_manager = BaseEnvManager  # for compatibility
        env_config.manager = deep_merge_dicts(env_manager.default_config(), env_config.manager)
        policy_config = policy.default_config()
        policy_config = deep_merge_dicts(policy_config_template, policy_config)
        if have_adversary:
            policy_adversary_config = policy_adversary.default_config()
            policy_adversary_config = deep_merge_dicts(policy_config_template, policy_adversary_config)
        if reward_model is None:
            reward_model_config = EasyDict()
        else:
            reward_model_config = reward_model.default_config()
        if world_model is None:
            world_model_config = EasyDict()
        else:
            world_model_config = world_model.default_config()
            world_model_config.update(create_cfg.world_model)

    policy_config.learn.learner = deep_merge_dicts(
        learner.default_config(),
        policy_config.learn.learner,
    )
    if have_adversary:
        policy_adversary_config.learn.learner = deep_merge_dicts(
            learner.default_config(),
            policy_adversary_config.learn.learner,
        )
    if create_cfg is not None or collector is not None:
        policy_config.collect.collector = compile_collector_config(policy_config, cfg, collector)
        if have_adversary:
            policy_adversary_config.collect.collector = compile_collector_config(policy_adversary_config, cfg, collector)
    if evaluator:
        policy_config.eval.evaluator = deep_merge_dicts(
            evaluator.default_config(),
            policy_config.eval.evaluator,
        )
        if have_adversary:
            policy_adversary_config.eval.evaluator = deep_merge_dicts(
                evaluator.default_config(),
                policy_adversary_config.eval.evaluator,
            )
    if create_cfg is not None or buffer is not None:
        policy_config.other.replay_buffer = compile_buffer_config(policy_config, cfg, buffer)
        if have_adversary:
            policy_adversary_config.other.replay_buffer = compile_buffer_config(policy_adversary_config, cfg, buffer)

    default_config = EasyDict({'env': env_config, 'policy': policy_config, 'policy_adversary': policy_adversary_config})
    if len(reward_model_config) > 0:
        default_config['reward_model'] = reward_model_config
    if len(world_model_config) > 0:
        default_config['world_model'] = world_model_config
    cfg = deep_merge_dicts(default_config, cfg)
    if 'unroll_len' in cfg.policy:
        cfg.policy.collect.unroll_len = cfg.policy.unroll_len
    if have_adversary:
        if 'unroll_len' in cfg.policy_adversary:
            cfg.policy_adversary.collect.unroll_len = cfg.policy_adversary.unroll_len
    cfg.seed = seed
    # check important key in config
    if evaluator in [InteractionSerialEvaluator, BattleInteractionSerialEvaluator]:  # env interaction evaluation
        cfg.policy.eval.evaluator.stop_value = cfg.env.stop_value
        cfg.policy.eval.evaluator.n_episode = cfg.env.n_evaluator_episode
        if have_adversary:
            cfg.policy_adversary.eval.evaluator.stop_value = cfg.env.stop_value
            cfg.policy_adversary.eval.evaluator.n_episode = cfg.env.n_evaluator_episode
    if 'exp_name' not in cfg:
        cfg.exp_name = 'default_experiment'
    if save_cfg and get_rank() == 0:
        if os.path.exists(cfg.exp_name) and renew_dir:
            cfg.exp_name += datetime.datetime.now().strftime("_%y%m%d_%H%M%S")
        try:
            os.makedirs(cfg.exp_name)
        except FileExistsError:
            pass
        save_project_state(cfg.exp_name)
        save_path = os.path.join(cfg.exp_name, save_path)
        save_config(cfg, save_path, save_formatted=True)
    return cfg