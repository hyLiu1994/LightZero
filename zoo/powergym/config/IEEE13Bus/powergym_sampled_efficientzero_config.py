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


ignore_done = False
weight_decay = 5e-7

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


max_env_step = int(5e8)
reanalyze_ratio = 0.
policy_entropy_loss_weight = 0.005
eval_freq = 1000
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

powergym_sampled_efficientzero_config = dict(
    exp_name=
f'data_sez_ctree_IEEE13/IEEE13_{K}_{env_id}_sampled_efficientZero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_bs-{batch_size}_pelw{policy_entropy_loss_weight}_seed{seed}',
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
        weight_decay=weight_decay,  # 0.01 不太行
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(eval_freq),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
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
        type='sampled_efficientzero',
        import_names=['lzero.policy.sampled_efficientzero'],
    ),

)
powergym_sampled_efficientzero_create_config = EasyDict(powergym_sampled_efficientzero_create_config)
create_config = powergym_sampled_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=seed, max_env_step=max_env_step)


