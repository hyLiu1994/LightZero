import numpy as np
import os
import copy
import random
import argparse
import json
import torch
import logging
logging.disable(logging.INFO)

from .policy_gradients import models
from .policy_gradients.apply_ppo_adversary_agent import Trainer

def apply(params):
    override_params = copy.deepcopy(params)
    excluded_params = ['config_path', 'out_dir_prefix', 'num_episodes', 'row_id', 'exp_id', 'num_actions', 'num_features', 'agent_type',
            'load_model', 'seed', 'deterministic', 'noise_factor', 'compute_kl_cert', 'use_full_backward', 'sqlite_path', 'early_terminate']

    # original_params contains all flags in config files that are overridden via command.
    for k in list(override_params.keys()):
        if k in excluded_params:
            del override_params[k]

    # Append a prefix for output path.
    if params['out_dir_prefix']:
        params['out_dir'] = os.path.join(params['out_dir_prefix'], params['out_dir'])
        print(f"setting output dir to {params['out_dir']}")

    if params['config_path']:
        # Load from a pretrained model using existing config.
        # First we need to create the model using the given config file.
        with open(params['config_path']) as file:
            json_params = json.load(file)
        
        params = override_json_params(params, json_params, excluded_params)

    for k, v in zip(params.keys(), params.values()):
        assert v is not None, f"Value for {k} is None"

    p = Trainer.agent_from_params(params, store=None)

    if 'load_model' in params and params['load_model']:
        print('Loading pretrained model', params['load_model'])
        pretrained_model = torch.load(params['load_model'])
        if 'policy_model' in pretrained_model:
            p.policy_model.load_state_dict(pretrained_model['policy_model'])
        if 'val_model' in pretrained_model:
            p.val_model.load_state_dict(pretrained_model['val_model'])
        if 'policy_opt' in pretrained_model:
            p.POLICY_ADAM.load_state_dict(pretrained_model['policy_opt'])
        if 'val_opt' in pretrained_model:
            p.val_opt.load_state_dict(pretrained_model['val_opt'])
        # Restore environment parameters, like mean and std.
        if 'envs' in pretrained_model:
            p.envs = pretrained_model['envs']
        for e in p.envs:
            e.normalizer_read_only = True
            e.setup_visualization(params['show_env'], params['save_frames'], params['save_frames_path'])

    return p

def override_json_params(params, json_params, excluding_params):
    # Override the JSON config with the argparse config
    missing_keys = []
    for key in json_params:
        if key not in params:
            missing_keys.append(key)
    assert not missing_keys, "Following keys not in args: " + str(missing_keys)

    missing_keys = []
    for key in params:
        if key not in json_params and key not in excluding_params:
            missing_keys.append(key)
    assert not missing_keys, "Following keys not in JSON: " + str(missing_keys)

    json_params.update({k: params[k] for k in params if params[k] is not None})
    return json_params

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_common_parser_opts(parser):
    # Basic setup
    parser.add_argument('--game', type=str, help='gym game')
    parser.add_argument('--mode', type=str, choices=['ppo', 'trpo', 'robust_ppo', 'adv_ppo', 'adv_trpo', 'adv_sa_ppo'],
                        help='pg alg')
    parser.add_argument('--out-dir', type=str,
                        help='out dir for store + logging')
    parser.add_argument('--advanced-logging', type=str2bool, const=True, nargs='?')
    parser.add_argument('--kl-approximation-iters', type=int,
                        help='how often to do kl approx exps')
    parser.add_argument('--log-every', type=int)
    parser.add_argument('--policy-net-type', type=str,
                        choices=models.POLICY_NETS.keys())
    parser.add_argument('--value-net-type', type=str,
                        choices=models.VALUE_NETS.keys())
    parser.add_argument('--train-steps', type=int,
                        help='num agent training steps')
    parser.add_argument('--cpu', type=str2bool, const=True, nargs='?')

    # Which value loss to use
    parser.add_argument('--value-calc', type=str,
                        help='which value calculation to use')
    parser.add_argument('--initialization', type=str)

    # General Policy Gradient parameters
    parser.add_argument('--num-actors', type=int, help='num actors (serial)',
                        choices=[1])
    parser.add_argument('--t', type=int,
                        help='num timesteps to run each actor for')
    parser.add_argument('--gamma', type=float, help='discount on reward')
    parser.add_argument('--lambda', type=float, help='GAE hyperparameter')
    parser.add_argument('--val-lr', type=float, help='value fn learning rate')
    parser.add_argument('--val-epochs', type=int, help='value fn epochs')
    parser.add_argument('--initial-std', type=float, help='initial value of std for Gaussian policy. Default is 1.')

    # PPO parameters
    parser.add_argument('--adam-eps', type=float, choices=[0, 1e-5], help='adam eps parameter')

    parser.add_argument('--num-minibatches', type=int,
                        help='num minibatches in ppo per epoch')
    parser.add_argument('--ppo-epochs', type=int)
    parser.add_argument('--ppo-lr', type=float,
                        help='if nonzero, use gradient descent w this lr')
    parser.add_argument('--ppo-lr-adam', type=float,
                        help='if nonzero, use adam with this lr')
    parser.add_argument('--anneal-lr', type=str2bool,
                        help='if we should anneal lr linearly from start to finish')
    parser.add_argument('--clip-eps', type=float, help='ppo clipping')
    parser.add_argument('--clip-val-eps', type=float, help='ppo clipping value')
    parser.add_argument('--entropy-coeff', type=float,
                        help='entropy weight hyperparam')
    parser.add_argument('--value-clipping', type=str2bool,
                        help='should clip values (w/ ppo eps)')
    parser.add_argument('--value-multiplier', type=float,
                        help='coeff for value loss in combined step ppo loss')
    parser.add_argument('--share-weights', type=str2bool,
                        help='share weights in valnet and polnet')
    parser.add_argument('--clip-grad-norm', type=float,
                        help='gradient norm clipping (-1 for no clipping)')
    parser.add_argument('--policy-activation', type=str,
                        help='activation function for countinous policy network')

    # TRPO parameters
    parser.add_argument('--max-kl', type=float, help='trpo max kl hparam')
    parser.add_argument('--max-kl-final', type=float, help='trpo max kl final')
    parser.add_argument('--fisher-frac-samples', type=float,
                        help='frac samples to use in fisher vp estimate')
    parser.add_argument('--cg-steps', type=int,
                        help='num cg steps in fisher vp estimate')
    parser.add_argument('--damping', type=float, help='damping to use in cg')
    parser.add_argument('--max-backtrack', type=int, help='max bt steps in fvp')
    parser.add_argument('--trpo-kl-reduce-func', type=str,
                        help='reduce function for KL divergence used in line search. mean or max.')

    # Robust PPO parameters.
    parser.add_argument('--robust-ppo-eps', type=float, help='max eps for robust PPO training')
    parser.add_argument('--robust-ppo-method', type=str, choices=['convex-relax', 'sgld', 'pgd'],
                        help='robustness regularization methods')
    parser.add_argument('--robust-ppo-pgd-steps', type=int, help='number of PGD optimization steps')
    parser.add_argument('--robust-ppo-detach-stdev', type=str2bool, help='detach gradient of standard deviation term')
    parser.add_argument('--robust-ppo-reg', type=float, help='robust PPO regularization')
    parser.add_argument('--robust-ppo-eps-scheduler-opts', type=str,
                        help='options for epsilon scheduler for robust PPO training')
    parser.add_argument('--robust-ppo-beta', type=float, help='max beta (IBP mixing factor) for robust PPO training')
    parser.add_argument('--robust-ppo-beta-scheduler-opts', type=str,
                        help='options for beta scheduler for robust PPO training')

    # Adversarial PPO parameters.
    parser.add_argument('--adv-ppo-lr-adam', type=float,
                        help='if nonzero, use adam for adversary policy with this lr')
    parser.add_argument('--adv-entropy-coeff', type=float,
                        help='entropy weight hyperparam for adversary policy')
    parser.add_argument('--adv-eps', type=float, help='adversary perturbation eps')
    parser.add_argument('--adv-clip-eps', type=float, help='ppo clipping for adversary policy')
    parser.add_argument('--adv-val-lr', type=float, help='value fn learning rate for adversary policy')
    parser.add_argument('--adv-policy-steps', type=float, help='number of policy steps before adversary steps')
    parser.add_argument('--adv-adversary-steps', type=float, help='number of adversary steps before adversary steps')
    parser.add_argument('--adv-adversary-ratio', type=float, help='percentage of frames to attack for the adversary')

    # Adversarial attack parameters.
    parser.add_argument('--attack-method', type=str,
                        choices=["none", "critic", "random", "action", "sarsa", "sarsa+action", "advpolicy",
                                 "action+imit"], help='adversarial attack methods.')
    parser.add_argument('--attack-ratio', type=float, help='attack only a ratio of steps.')
    parser.add_argument('--attack-steps', type=int, help='number of PGD optimization steps.')
    parser.add_argument('--attack-eps', type=str,
                        help='epsilon for attack. If set to "same", we will use value of robust-ppo-eps.')
    parser.add_argument('--attack-step-eps', type=str,
                        help='step size for each iteration. If set to "auto", we will use attack-eps / attack-steps')
    parser.add_argument('--attack-sarsa-network', type=str, help='sarsa network to load for attack.')
    parser.add_argument('--attack-sarsa-action-ratio', type=float,
                        help='When set to non-zero, enable sarsa-action attack.')
    parser.add_argument('--attack-advpolicy-network', type=str, help='adversarial policy network to load for attack.')
    parser.add_argument('--collect-perturbed-states', type=str2bool, help='collect perturbed states during training')

    # Normalization parameters
    parser.add_argument('--norm-rewards', type=str, help='type of rewards normalization',
                        choices=['rewards', 'returns', 'none'])
    parser.add_argument('--norm-states', type=str2bool, help='should norm states')
    parser.add_argument('--clip-rewards', type=float, help='clip rews eps')
    parser.add_argument('--clip-observations', type=float, help='clips obs eps')

    # Sequence training parameters
    parser.add_argument('--history-length', type=int,
                        help='length of history to use for LSTM. If <= 1, we do not use LSTM.')
    parser.add_argument('--use-lstm-val', type=str2bool, help='use a lstm for value function')

    # Saving
    parser.add_argument('--save-iters', type=int, help='how often to save model (0 = no saving)')
    parser.add_argument('--force-stop-step', type=int,
                        help='forcibly terminate after a given number of steps. Useful for debugging and tuning.')

    # Visualization
    parser.add_argument('--show-env', type=str2bool, help='Show environment visualization')
    parser.add_argument('--save-frames', type=str2bool, help='Save environment frames')
    parser.add_argument('--save-frames-path', type=str, help='Path to save environment frames')

    # For grid searches only
    # parser.add_argument('--cox-experiment-path', type=str, default='')
    return parser

def get_parser():
    parser = argparse.ArgumentParser(description='Generate experiments to be run.')
    parser.add_argument('--config-path', type=str, default='', required=False,
                        help='json for this config')
    parser.add_argument('--out-dir-prefix', type=str, default='', required=False,
                        help='prefix for output log path')
    parser.add_argument('--exp-id', type=str, help='experiement id for testing', default='')
    parser.add_argument('--row-id', type=int, help='which row of the table to use', default=-1)
    parser.add_argument('--num-episodes', type=int, help='number of episodes for testing', default=50)
    parser.add_argument('--compute-kl-cert', action='store_true', help='compute KL certificate')
    parser.add_argument('--use-full-backward', action='store_true', help='Use full backward LiRPA bound for computing certificates')
    parser.add_argument('--deterministic', action='store_true', help='disable Gaussian noise in action for evaluation')
    parser.add_argument('--noise-factor', type=float, default=1.0, help='increase the noise (Gaussian std) by this factor.')
    parser.add_argument('--load-model', type=str, help='load a pretrained model file', default='')
    parser.add_argument('--seed', type=int, help='random seed', default=1234)

    parser.add_argument('--num-actions', type=int, help='env actions', default=3)
    parser.add_argument('--num-features', type=int, help='env features', default=11)
    parser.add_argument('--agent-type', type=str, help='env actions type', default='continuous') # discrete

    parser = add_common_parser_opts(parser)
    
    return parser


def main(seed, ppo_adv_config_path, attack_method, attack_advpolicy_network,
         num_actions, num_features, agent_type):
    parser = get_parser()
    args = parser.parse_args()
    if args.load_model:
        assert args.config_path, "Need to specificy a config file when loading a pretrained model."

    params = vars(args)
    # seed = params['seed']
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    params['config_path'] = ppo_adv_config_path
    params['attack_method'] = attack_method
    params['attack_advpolicy_network'] = attack_advpolicy_network
    params['num_actions'] = num_actions
    params['num_features'] = num_features
    params['agent_type'] = agent_type

 
    return apply(params)

