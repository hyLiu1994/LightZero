from easydict import EasyDict

# options={'Hopper-v2', 'HalfCheetah-v3', 'Walker2d-v3', 'Ant-v3', 'Humanoid-v3'}
env_id = 'Hopper-v2'

if env_id == 'Hopper-v2':
    action_space_size = 3
    observation_shape = 11
elif env_id in ['HalfCheetah-v3', 'Walker2d-v3']:
    action_space_size = 6
    observation_shape = 17
elif env_id == 'Ant-v3':
    action_space_size = 8
    observation_shape = 111
elif env_id == 'Humanoid-v3':
    action_space_size = 17
    observation_shape = 376

ignore_done = False
if env_id == 'HalfCheetah-v3':
    # for halfcheetah, we ignore done signal to predict the Q value of the last step correctly.
    ignore_done = True

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
seed = 0
n_episode = 8
collector_env_num = 8
evaluator_env_num = 5
continuous_action_space = True
K = 50  # num_of_sampled_actions  50 125
num_simulations = 125
update_per_collect = 200
batch_size = 256
Epsilon = 0.075


max_env_step = int(5e8)
reanalyze_ratio = 0.
policy_entropy_loss_weight = 0.005
eval_freq = 50
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

mujoco_sampled_efficientzero_config = dict(
    exp_name=
f'data_sez_ctree_pytest/{env_id[:-3]}_Muzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_bs-{batch_size}_pelw{policy_entropy_loss_weight}_seed{seed}',
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
            latent_state_dim=256,
            self_supervised_learning_loss=True,
            res_connection_in_dynamics=True,
        ),
        cuda=True,
        policy_entropy_loss_weight=policy_entropy_loss_weight,
        ignore_done=ignore_done,
        env_type='not_board_games',
        game_segment_length=200,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        discount_factor=0.997,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        grad_clip_value=0.5,  # 需要小点
        weight_decay=5e-6,  # 0.01 不太行
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=eval_freq,
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        noise_policy = 'normal'
    ),
    policy_adversary=dict(
        action_space='continuous',
        # -----------------------------------------------
        obs_shape=observation_shape,
        action_shape=action_space_size,
        env_seed=seed,
        ppo_adv_config_path='/root/autodl-tmp/LightZero/ATLA_robust_RL/src/config_hopper_atla_ppo.json',
        attack_method='advpolicy',
        attack_advpolicy_network='/root/autodl-tmp/LightZero/ATLA_robust_RL/src/models/atla_release/ATLA-PPO/attack-atla-ppo-hopper.model',
        Epsilon=Epsilon,
        cuda=True,
        noise_policy='ppo',  # 'atla_ppo' 'ppo'
        #------------------------------------------------------------------------------
    ),
    policy_random_adversary=dict(
        cuda=True,
        Epsilon=Epsilon,
        noise_policy='random',
    ),

)

mujoco_sampled_efficientzero_config = EasyDict(mujoco_sampled_efficientzero_config)
main_config = mujoco_sampled_efficientzero_config

mujoco_sampled_efficientzero_create_config = dict(
    env=dict(
        type='mujoco_lightzero',
        import_names=['zoo.mujoco.envs.mujoco_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_efficientzero',
        import_names=['lzero.policy.sampled_efficientzero'],
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
mujoco_sampled_efficientzero_create_config = EasyDict(mujoco_sampled_efficientzero_create_config)
create_config = mujoco_sampled_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import eval_muzero_with_adversary
    eval_muzero_with_adversary([main_config, create_config], seed=seed, model_path='SampledZero-Mujoco/Hopper_sampled_efficientzero_ns125_upc200_rr0.0_bs-256_pelw0.005_seed0_240716_172345/ckpt/ckpt_best.pth.tar', num_episodes_each_seed=1, print_seed_details=True)


