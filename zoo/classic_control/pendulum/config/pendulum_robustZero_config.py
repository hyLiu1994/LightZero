from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
continuous_action_space = True
K = 20  # num_of_sampled_actions
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = 200
batch_size = 256
max_env_step = int(2e5)
reanalyze_ratio = 0.
eval_freq = 200
seed = 0
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

pendulum_sampled_efficientzero_config = dict(
    exp_name=
    f'data_sez_ctree_pendulum/pendulum_RobustZero_k{K}_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_bs_{batch_size}_seed_{seed}',
    env=dict(
        env_id='Pendulum-v1',
        continuous=True,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=3,
            action_space_size=11,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            sigma_type='conditioned',
            model_type='mlp', 
            lstm_hidden_size=128,
            latent_state_dim=128,
            self_supervised_learning_loss=True,
            self_supervised_adversary_learning_loss=True,
        ),
        # RobustZero hyperparamter ------
        c3=0.5,
        c4=1,
        robustzero_w1=1,
        optim_type='AdamAd',
        robustzero_lambda=0.0001,  # 0.0015  5e-6    5e-5 0.015 0.001
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
        # NOTE: for continuous gaussian policy, we use the policy_entropy_loss as in the original Sampled MuZero paper.
        policy_entropy_loss_weight=5e-3,
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
        obs_shape=3,
        action_shape=11,
        env_seed=seed,
        attack_method='advpolicy',
        ppo_adv_config_path='/root/autodl-tmp/LightZero/ATLA_robust_RL/src/config_pendulum_atla_ppo.json',
        attack_advpolicy_network='/root/autodl-tmp/LightZero/ATLA_robust_RL/src/models/atla_release/ATLA-PPO/attack-atla-ppo-pendulum.model',
        Epsilon=0.075,
        noise_policy='ppo',  # 'atla_ppo' 'ppo'
        # ---------------------------------------------------------------------
    ),
    policy_random_adversary=dict(
        Epsilon=0.075,
        noise_policy='random',
    ),

)
pendulum_sampled_efficientzero_config = EasyDict(pendulum_sampled_efficientzero_config)
main_config = pendulum_sampled_efficientzero_config

pendulum_sampled_efficientzero_create_config = dict(
    env=dict(
        type='pendulum_lightzero',
        import_names=['zoo.classic_control.pendulum.envs.pendulum_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='robustzero',
        import_names=['lzero.policy.robustzero'],
    ),
    # collector=dict(
    #     type='episode_muzero',
    #     get_train_sample=True,
    #     import_names=['lzero.worker.muzero_collector'],
    # )
)
pendulum_sampled_efficientzero_create_config = EasyDict(pendulum_sampled_efficientzero_create_config)
create_config = pendulum_sampled_efficientzero_create_config

if __name__ == "__main__":
    # Users can use different train entry by specifying the entry_type.
    from lzero.entry import train_robustzero
    train_robustzero([main_config, create_config], seed=seed, max_env_step=max_env_step)
