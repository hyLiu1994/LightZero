from easydict import EasyDict
import zoo.powergym.env_manager.power_gym_subprocess_env_manager
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# options={'13Bus', '34Bus', '123Bus', '8500-Node'}
env_id = '123Bus'

if env_id == '13Bus':
    action_space_size = 6
    observation_shape = 48
elif env_id == '34Bus':
    action_space_size = 10
    observation_shape = 107
elif env_id == '123Bus':
    action_space_size = 15
    observation_shape = 297

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
continuous_action_space = True
K = 20  # num_of_sampled_actions
collector_env_num = 3
n_episode = 3
evaluator_env_num = 2
num_simulations = 50
update_per_collect = None
replay_ratio = 0.25
batch_size = 256
max_env_step = int(2e5)
reanalyze_ratio = 0.
norm_type = 'LN'
seed = 0
eval_freq = 1000
latent_state_dim = 256
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

powergym_sampled_muzero_config = dict(
    exp_name=f'data_smz/{env_id}_sampled_muzero_k{K}_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_norm-{norm_type}_seed_{seed}',
    env=dict(
        env_id=env_id,
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
            sigma_type='conditioned',
            model_type='mlp', 
            latent_state_dim=latent_state_dim,
            norm_type=norm_type,
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=200,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
        cos_lr_scheduler=True,
        learning_rate=0.0001,
        lr_piecewise_constant_decay=False,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(eval_freq),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        ssl_loss_weight=0.5
    ),
)
powergym_sampled_muzero_config = EasyDict(powergym_sampled_muzero_config)
main_config = powergym_sampled_muzero_config

powergym_sampled_muzero_create_config = dict(
    env=dict(
        type='powergym_lightzero',
        import_names=['zoo.powergym.envs.powergym_lightzero_env'],
    ),
    env_manager=dict(type='power_gym_subprocess'),
    policy=dict(
        type='sampled_gumbel_efficientzero',
        import_names=['lzero.policy.sampled_gumbel_efficientzero'],
    ),
)
powergym_sampled_muzero_create_config = EasyDict(powergym_sampled_muzero_create_config)
create_config = powergym_sampled_muzero_create_config

if __name__ == "__main__":
    from lzero.entry.train_muzero import train_muzero
    train_muzero([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)
