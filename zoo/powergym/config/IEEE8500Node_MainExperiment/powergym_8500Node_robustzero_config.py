from easydict import EasyDict
import zoo.powergym.env_manager.power_gym_subprocess_env_manager
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
# parent_dir = os.path.dirname(parent_dir)
print(parent_dir)
sys.path.append(parent_dir)

# options={'13Bus', '34Bus', '123Bus', '8500-Node'}
env_id = '8500Node'

if env_id == '13Bus':
    action_space_size = 6
    observation_shape = 48
elif env_id == '34Bus':
    action_space_size = 10
    observation_shape = 107
elif env_id == '123Bus':
    action_space_size = 15
    observation_shape = 297
elif env_id == '8500Node':
    action_space_size = 32
    observation_shape = 8573

ignore_done = False
weight_decay = 1e-7

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
seed = 0
n_episode = 3
collector_env_num = 3 # 不要变动,若要变动, 只能往小值变小.
evaluator_env_num = 2
continuous_action_space = True
K = 50  # num_of_sampled_actions
num_simulations = 125
update_per_collect = 200
batch_size = 256

max_env_step = int(2e5)
reanalyze_ratio = 0.
policy_entropy_loss_weight = 0.005
eval_freq = 200
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

powergym_robustzero_config = dict(
    exp_name=
    f'data_sez_ctree_{env_id}/{K}_{env_id}_RobustZero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_bs-{batch_size}_pelw{policy_entropy_loss_weight}_seed{seed}_wd{weight_decay}',
    env=dict(
        env_id=env_id,
        action_clip=True,
        continuous=True,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=observation_shape,
            action_space_size=action_space_size,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            model_type='mlp',
            lstm_hidden_size=256,
            latent_state_dim=observation_shape,
            self_supervised_learning_loss=True,
            self_supervised_adversary_learning_loss=True,
            res_connection_in_dynamics=True,
        ),
        cuda=True,
        # RobustZero hyperparamter ------
        c3=0.5,
        c4=1,
        robustzero_w1=-1,
        optim_type='AdamAd',
        robustzero_lambda=weight_decay,
        # -------------------------------
        policy_entropy_loss_weight=policy_entropy_loss_weight,
        ignore_done=ignore_done,
        env_type='not_board_games',
        game_segment_length=200,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        discount_factor=0.997,
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        # optim_type='Adam',
        weight_decay=weight_decay,  # 0.01 不太行
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(eval_freq),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
    policy_adversary=dict(
        action_space='continuous',
        obs_shape=observation_shape,
        action_shape=action_space_size,
        env_seed=seed,
        attack_method='advpolicy',
        ppo_adv_config_path=f'/root/autodl-tmp/LightZero/ATLA_robust_RL/src/config_{env_id}_atla_ppo.json',
        attack_advpolicy_network=f'/root/autodl-tmp/LightZero/ATLA_robust_RL/src/models/atla_release/ATLA-PPO/attack-atla-ppo-{env_id}-eps0.15-no-norm.model',
        Epsilon=0.15,
        noise_policy='ppo',  # 'atla_ppo' 'ppo'
        # ------------------------------------------------------------------------------
    ),
    policy_random_adversary=dict(
        Epsilon=0.15,
        noise_policy='random',
    ),
)

powergym_robustzero_config = EasyDict(powergym_robustzero_config)
main_config = powergym_robustzero_config

powergym_robustzero_create_config = dict(
    env=dict(
        type='powergym_lightzero',
        import_names=['zoo.powergym.envs.powergym_lightzero_env'],
    ),
    env_manager=dict(type='power_gym_subprocess'),
    policy=dict(
        type='robustzero',
        import_names=['lzero.policy.robustzero'],
        # learner=dict(
        #     train_iterations=int(1e4),
        #     dataloader=dict(num_workers=0, ),
        #     log_policy=True,
        #     hook=dict(
        #         load_ckpt_before_run='',
        #         log_show_after_iter=100,
        #         save_ckpt_after_iter=10000,
        #         save_ckpt_after_run=True,
        #     ),
        # ),
    ),
    # policy_adversary=dict(type='ppo'),
)
powergym_robustzero_create_config = EasyDict(powergym_robustzero_create_config)
create_config = powergym_robustzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_robustzero
    train_robustzero([main_config, create_config], seed=seed, max_env_step=max_env_step)


