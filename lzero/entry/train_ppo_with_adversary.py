import logging
import os
from functools import partial
from typing import Optional, Tuple
from copy import deepcopy

import torch
from ding.envs import create_env_manager
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed, get_rank
from ding.rl_utils import get_epsilon_greedy_fn
from tensorboardX import SummaryWriter

from lzero.config.compile_config import compile_config
from ding.worker import BaseLearner,  BaseSerialCommander
from lzero.entry.utils import log_buffer_memory_usage
from lzero.policy import visit_count_temperature
from lzero.worker import PPOSampleSerialCollector as Collector
from lzero.worker import PPOInteractionSerialEvaluator as Evaluator
from lzero.worker import InteractionAdversarySerialEvaluator as EvaluatorAdversary
from lzero.worker import AdversarySampleSerialCollector as CollectorAdversary



def train_ppo_with_adversary(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        The train entry for MCTS+RL algorithms, including MuZero, EfficientZero, Sampled EfficientZero, Gumbel Muzero.
    Arguments:
        - input_cfg (:obj:`Tuple[dict, dict]`): Config in dict type.
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - model_path (:obj:`Optional[str]`): The pretrained model path, which should
            point to the ckpt file of the pretrained model, and an absolute path is recommended.
            In LightZero, the path is usually something like ``exp_name/ckpt/ckpt_best.pth.tar``.
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
        - max_env_step (:obj:`Optional[int]`): Maximum collected environment interaction steps.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """

    cfg, create_cfg = input_cfg
    assert create_cfg.policy.type in [ 'sampled_ppo'], \
        ("train_muzero entry now only support the following algo.: 'sampled_ppo' ")

    # sampled_ppo 不需要buffer
    # if create_cfg.policy.type == 'muzero':
    #     from lzero.mcts import MuZeroGameBuffer as GameBuffer
    # elif create_cfg.policy.type == 'efficientzero':
    #     from lzero.mcts import EfficientZeroGameBuffer as GameBuffer
    # elif create_cfg.policy.type == 'sampled_efficientzero':
    #     from lzero.mcts import SampledEfficientZeroGameBuffer as GameBuffer
    # elif create_cfg.policy.type == 'sampled_adversary_efficientzero':
    #     from lzero.mcts import AdversarySampledEfficientZeroGameBuffer as GameBuffer
    # elif create_cfg.policy.type == 'sampled_two_adversary_efficientzero':
    #     from lzero.mcts import AdversarySampledEfficientZeroGameBuffer as GameBuffer
    # elif create_cfg.policy.type == 'gumbel_muzero':
    #     from lzero.mcts import GumbelMuZeroGameBuffer as GameBuffer
    # elif create_cfg.policy.type == 'stochastic_muzero':
    #     from lzero.mcts import StochasticMuZeroGameBuffer as GameBuffer

    if cfg.policy.cuda and torch.cuda.is_available():
        cfg.policy.device = 'cuda:0'
    else:
        cfg.policy.device = 'cpu'


    create_cfg.policy_adversary.type = create_cfg.policy_adversary.type + '_command'
    # create_cfg.policy.type = create_cfg.policy.type + '_command'
    policy_adversary_config = cfg.policy_adversary
    policy_random_adversary_config = cfg.policy_random_adversary

    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True, have_adversary=True)
    # Create main components: env, policy
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)


    normal_evaluator_env_cfg = deepcopy(evaluator_env_cfg)
    [ne.__setattr__('env_type', 'normal_evaluator') for ne in normal_evaluator_env_cfg]

    ppo_collector_env_cfg = deepcopy(collector_env_cfg)
    [pc.__setattr__('env_type', 'ppo_collector') for pc in ppo_collector_env_cfg]
    ppo_evaluator_env_cfg = deepcopy(evaluator_env_cfg)
    [pe.__setattr__('env_type', 'ppo_evaluator') for pe in ppo_evaluator_env_cfg]


    random_evaluator_env_cfg = deepcopy(evaluator_env_cfg)
    [re.__setattr__('env_type', 'random_evaluator') for re in random_evaluator_env_cfg]

    ppo_adversary_collector_env_cfg = deepcopy(collector_env_cfg)
    [pca.__setattr__('env_type', 'ppo_adversary_collector') for pca in ppo_adversary_collector_env_cfg]
    ppo_adversary_evaluator_env_cfg = deepcopy(evaluator_env_cfg)
    [pea.__setattr__('env_type', 'ppo_adversary_evaluator') for pea in ppo_adversary_evaluator_env_cfg]


    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in normal_evaluator_env_cfg])
    evaluator_env.seed(cfg.seed, dynamic_seed=False)


    ppo_collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in ppo_collector_env_cfg])
    ppo_collector_env.seed(cfg.seed)
    ppo_evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in ppo_evaluator_env_cfg])
    ppo_evaluator_env.seed(cfg.seed, dynamic_seed=False)


    random_evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in random_evaluator_env_cfg])
    random_evaluator_env.seed(cfg.seed, dynamic_seed=False)

    collector_adversary_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in ppo_adversary_collector_env_cfg])
    evaluator_adversary_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in ppo_adversary_evaluator_env_cfg])
    collector_adversary_env.seed(cfg.seed)
    evaluator_adversary_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])
    policy_adversary = create_policy(cfg.policy_adversary, model=model, enable_field=['learn', 'collect', 'eval', 'command'])
    # policy_random_adversary = create_policy(cfg.policy_random_adversary, model=model, enable_field=['collect', 'eval'])

    # load pretrained model
    if model_path is not None:
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, instance_name='agent_learner', exp_name=cfg.exp_name)

    learner_adversary = BaseLearner(cfg.policy_adversary.learn.learner, policy_adversary.learn_mode,
                                    tb_logger, instance_name='adversary_learner', exp_name=cfg.exp_name)

    # ==============================================================
    # MCTS+RL algorithms related core code
    # ==============================================================
    policy_config = cfg.policy
    # batch_size = policy_config.batch_size
    # # specific game buffer for MCTS+RL algorithms
    # replay_buffer = GameBuffer(policy_config)

    evaluator = Evaluator(
        cfg=cfg.policy.eval.evaluator,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        instance_name="agent_evaluator"
    )

    collector = Collector(
        cfg.policy.collect.collector,
        env=ppo_collector_env,
        policy=policy.collect_mode,
        policy_adversary=policy_adversary.collect_mode,
        policy_adversary_config=policy_adversary_config,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        instance_name="agent_collector_with_ppo"
    )
    ppo_evaluator = Evaluator(
        cfg=cfg.policy.eval.evaluator,
        env=ppo_evaluator_env,
        policy=policy.eval_mode,
        policy_adversary=policy_adversary.eval_mode,
        policy_adversary_config=policy_adversary_config,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        instance_name = 'agent_evaluator_with_ppo',
    )
    random_evaluator = Evaluator(
        cfg=cfg.policy.eval.evaluator,
        env=random_evaluator_env,
        policy=policy.eval_mode,
        policy_adversary=None,
        policy_adversary_config=policy_random_adversary_config,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        instance_name='agent_evaluator_with_random',
     )

    collector_adversary = CollectorAdversary(
        cfg.policy_adversary.collect.collector,
        env=collector_adversary_env,
        policy=policy_adversary.collect_mode,
        policy_agent=policy.eval_mode,
        policy_config=policy_adversary_config,
        policy_agent_config=policy_config,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        instance_name='adversary_collector',
    )
    evaluator_adversary = EvaluatorAdversary(
        cfg.policy_adversary.eval.evaluator,
        env = evaluator_adversary_env,
        policy = policy_adversary.eval_mode,
        policy_agent = policy.eval_mode,
        policy_config = policy_adversary_config,
        policy_agent_config = policy_config,
        tb_logger = tb_logger,
        exp_name=cfg.exp_name,
        instance_name='adversary_evaluator',
    )
    commander = BaseSerialCommander(
        cfg.policy_adversary.other.commander,
        learner_adversary,
        collector_adversary,
        evaluator_adversary,
        None,
        policy_adversary.command_mode
    )

    # ==============================================================
    # Main loop
    # ==============================================================
    # Learner's before_run hook.
    learner.call_hook('before_run')
    learner_adversary.call_hook('before_run')
    
    # if cfg.policy.update_per_collect is not None:
    #     update_per_collect = cfg.policy.update_per_collect

    # The purpose of collecting random data before training:
    # Exploration: Collecting random data helps the agent explore the environment and avoid getting stuck in a suboptimal policy prematurely.
    # Comparison: By observing the agent's performance during random action-taking, we can establish a baseline to evaluate the effectiveness of reinforcement learning algorithms.
    # if cfg.policy.random_collect_episode_num > 0:
    #     random_collect(cfg.policy, policy, LightZeroRandomPolicy, collector, collector_env, replay_buffer)

    while True:
        # log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)
        # collect_kwargs = {}
        # # set temperature for visit count distributions according to the train_iter,
        # # please refer to Appendix D in MuZero paper for details.
        # collect_kwargs['temperature'] = visit_count_temperature(
        #     policy_config.manual_temperature_decay,
        #     policy_config.fixed_temperature_value,
        #     policy_config.threshold_training_steps_for_final_temperature,
        #     trained_steps=learner.train_iter
        # )
        #
        # if policy_config.eps.eps_greedy_exploration_in_collect:
        #     epsilon_greedy_fn = get_epsilon_greedy_fn(
        #         start=policy_config.eps.start,
        #         end=policy_config.eps.end,
        #         decay=policy_config.eps.decay,
        #         type_=policy_config.eps.type
        #     )
        #     collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)
        # else:
        #     collect_kwargs['epsilon'] = 0.0

        print("Begin Evaluator!")
        # Evaluate policy performance.
        if evaluator.should_eval(learner.train_iter):
            stop, _ = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break

        print("Begin PPO Evaluator!")
        if ppo_evaluator.should_eval(learner.train_iter):
            stop, _ = ppo_evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break

        print("Begin Random Evaluator!")
        if random_evaluator.should_eval(learner.train_iter):
            stop, _ = random_evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break

        print("Begin Collect!")
        # Collect data by default config n_sample/n_episode.
        new_data = collector.collect(train_iter=learner.train_iter)

        # if cfg.policy.update_per_collect is None:
        #     # update_per_collect is None, then update_per_collect is set to the number of collected transitions multiplied by the model_update_ratio.
        #     collected_transitions_num = sum([len(game_segment) for game_segment in new_data[0]])
        #     update_per_collect = int(collected_transitions_num * cfg.policy.model_update_ratio)
        # save returned new_data collected by the collector
        # replay_buffer.push_game_segments(new_data)
        # # remove the oldest data if the replay buffer is full.
        # replay_buffer.remove_oldest_data_to_fit()
        # print(new_data)
        print("Begin Learn!")
        learner.train(new_data, collector.envstep)
        # Learn policy from collected data.
        # for i in range(update_per_collect):
        #     # Learner will train ``update_per_collect`` times in one iteration.
        #     if replay_buffer.get_num_of_transitions() > batch_size:
        #         train_data = replay_buffer.sample(batch_size, policy)
        #     else:
        #         logging.warning(
        #             f'The data in replay_buffer is not sufficient to sample a mini-batch: '
        #             f'batch_size: {batch_size}, '
        #             f'{replay_buffer} '
        #             f'continue to collect now ....'
        #         )
        #         break
        #
        #     # The core train steps for MCTS+RL algorithms.
        #
        #
        #     # if cfg.policy.use_priority:
        #     #     replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

        # Collecting Data for Adversary.
        collect_adversary_kwargs = commander.step()
        # Evaluate policy performance
        print("Begin Adversary Evaluator!")
        if evaluator_adversary.should_eval(learner_adversary.train_iter):
            stop, eval_info = evaluator_adversary.eval(learner_adversary.save_checkpoint, 
                                                       learner_adversary.train_iter, collector_adversary.envstep)
            if stop:
                break

        print("Begin Adversary Collector!")
        # Collect data by default config n_sample/n_episode
        new_data = collector_adversary.collect(train_iter=learner_adversary.train_iter, policy_kwargs=collect_adversary_kwargs)

        print("Begin Adversary Learner!")
        # Learn policy from collected data
        learner_adversary.train(new_data, collector_adversary.envstep)

        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
    learner_adversary.call_hook('after_run')
    return policy
