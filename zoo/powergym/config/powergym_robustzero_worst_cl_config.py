from easydict import EasyDict
import zoo.powergym.env_manager.power_gym_subprocess_env_manager
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# options={'13Bus', '34Bus', '123Bus', '8500-Node'}
env_id = '13Bus'

if env_id == '13Bus':
    action_space_size = 6
    observation_shape = 48
elif env_id == '34Bus':
    action_space_size = 10
    observation_shape = 107

ignore_done = False

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
seed = 0
n_episode = 3
collector_env_num = 3 # 不要变动,若要变动, 只能往小值变小.
evaluator_env_num = 1
continuous_action_space = True
K = 20  # num_of_sampled_actions
num_simulations = 50
update_per_collect = 200
batch_size = 256
eval_freq = 50

max_env_step = int(5e6)
reanalyze_ratio = 0.
policy_entropy_loss_weight = 0.005

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

powergym_sampled_efficientzero_config = dict(
    exp_name=
f'data_sez_ctree/{env_id}_RobustZero_with_ppo_cl_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_bs-{batch_size}_pelw{policy_entropy_loss_weight}_seed{seed}',
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
        # ssl_adversary_loss_weight
        c3=0.5,
        # -------------------------
        cuda=True,
        policy_entropy_loss_weight=policy_entropy_loss_weight,
        ignore_done=ignore_done,
        env_type='not_board_games',
        game_segment_length=200,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        discount_factor=0.997,
        optim_type='AdamW',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(eval_freq),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        mcts_ctree=True,
        noise_policy = 'ppo'
    ),
    policy_adversary=dict(
        cuda=True,
        recompute_adv=True,
        action_space='continuous',
        Epsilon=0.0075,
        noise_policy='ppo',
        model=dict(
            obs_shape=observation_shape,
            action_shape=observation_shape,
            action_space='continuous',
        ),
        learn=dict(
            epoch_per_collect=1,
            update_per_collect=1,
            batch_size=batch_size,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.001,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
            # for onppo, when we recompute adv, we need the key done in data to split traj, so we must
            # use ignore_done=False here,
            # but when we add key traj_flag in data as the backup for key done, we could choose to use ignore_done=True
            # for halfcheetah, the length=1000
            ignore_done=False,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
        ),
        collect=dict(
            n_sample=batch_size,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=eval_freq, )),
    ),
    policy_random_adversary=dict(
        cuda=True,
        # recompute_adv=True,
        # action_space='continuous',
        Epsilon=0.0075,
        noise_policy='random',
        # model=dict(
        #     obs_shape=observation_shape,
        #     action_shape=observation_shape,
        #     action_space='continuous',
        # ),
        # collect=dict(
        #     n_sample=3200,
        #     unroll_len=1,
        #     discount_factor=0.99,
        #     gae_lambda=0.95,
        # ),
        # eval=dict(evaluator=dict(eval_freq=500, )),
    ),

)

powergym_sampled_efficientzero_config = EasyDict(powergym_sampled_efficientzero_config)
main_config = powergym_sampled_efficientzero_config

powergym_sampled_efficientzero_create_config = dict(
    env=dict(
        type='powergym_lightzero',
        import_names=['zoo.powergym.envs.powergym_lightzero_env'],
    ),
    env_manager=dict(type='power_gym_subprocess'),
    policy=dict(
        type='sampled_adversary_efficientzero',
        import_names=['lzero.policy.sampled_adversary_efficientzero'],
        learner=dict(
            train_iterations=int(1e4),
            dataloader=dict(num_workers=0, ),
            log_policy=True,
            hook=dict(
                load_ckpt_before_run='',
                log_show_after_iter=100,
                save_ckpt_after_iter=10000,
                save_ckpt_after_run=True,
            ),
        ),
    ),
    policy_adversary=dict(type='ppo'),
)
powergym_sampled_efficientzero_create_config = EasyDict(powergym_sampled_efficientzero_create_config)
create_config = powergym_sampled_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero_with_adversary
    train_muzero_with_adversary([main_config, create_config], seed=seed, max_env_step=max_env_step)


