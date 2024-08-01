from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
continuous_action_space = False
K = 2  # num_of_sampled_actions
num_simulations = 50
update_per_collect = 100
batch_size = 256
max_env_step = int(1e5)
reanalyze_ratio = 0.
eval_freq = 200
seed = 0
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

cartpole_sampled_efficientzero_config = dict(
    exp_name=
    f'data_sez_ctree_cartpole/cartpole_RobustZero_c3_0.75_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_bs_{batch_size}_seed_{seed}',
    env=dict(
        env_id='CartPole-v1',
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=4,
            action_space_size=2,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            model_type='mlp', 
            lstm_hidden_size=128,
            latent_state_dim=128,
            discrete_action_encoding_type='one_hot',
            norm_type='BN', 
            self_supervised_learning_loss=True,
            self_supervised_adversary_learning_loss=True,
        ),
        # RobustZero hyperparamter ------
        c3=0.75,
        c4=1,
        robustzero_w1 = -1,
        optim_type='AdamAd',
        robustzero_lambda=0.0001,  #  0.0015  5e-6    5e-5 0.015 0.001
        # -------------------------------
        cuda=True,
        env_type='not_board_games',
        game_segment_length=50,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        learning_rate=0.003,
        grad_clip_value=0.5,  # 需要小点
        weight_decay=5e-6,  # 0.01 不太行
        lr_piecewise_constant_decay=False,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=eval_freq,
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
    policy_adversary=dict(
        action_space='continuous',
        obs_shape=4,
        action_shape=2,
        env_seed=seed,
        attack_method='advpolicy',
        ppo_adv_config_path='/root/autodl-tmp/LightZero/ATLA_robust_RL/src/config_cartpole_atla_ppo.json',
        attack_advpolicy_network='/root/autodl-tmp/LightZero/ATLA_robust_RL/src/models/atla_release/ATLA-PPO/attack-atla-ppo-cartpole-eps0.15-no-norm.model',
        Epsilon=0.15,
        noise_policy='ppo',  # 'atla_ppo' 'ppo'
        # ---------------------------------------------------------------------
    ),
    policy_random_adversary=dict(
        Epsilon=0.15,
        noise_policy='random',
    ),
)

cartpole_sampled_efficientzero_config = EasyDict(cartpole_sampled_efficientzero_config)
main_config = cartpole_sampled_efficientzero_config


cartpole_sampled_efficientzero_create_config = dict(
    env=dict(
        type='cartpole_lightzero',
        import_names=['zoo.classic_control.cartpole.envs.cartpole_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='robustzero',
        import_names=['lzero.policy.robustzero'],
    ),
)
cartpole_sampled_efficientzero_create_config = EasyDict(cartpole_sampled_efficientzero_create_config)
create_config = cartpole_sampled_efficientzero_create_config

if __name__ == "__main__":
    # Users can use different train entry by specifying the entry_type.
    from lzero.entry import train_robustzero
    train_robustzero([main_config, create_config], seed=seed, max_env_step=max_env_step)
