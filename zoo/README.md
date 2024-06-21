
## Environment Versatility

- The following is a brief introduction to the environment supported by our zoo：

<details open><summary>Expand for full list</summary>

| No |                                             Environment                                             |                                                             Label                                                             |                                               Visualization                                                |                                                                                       Doc Links                                                                                        |
|:--:|:---------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| 1  | [board_games/tictactoe](https://github.com/opendilab/LightZero/tree/main/zoo/board_games/tictactoe) |                                ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                                |   ![original](https://github.com/opendilab/LightZero/tree/main/zoo/board_games/tictactoe/tictactoe.gif)    |                                                               [env tutorial](https://en.wikipedia.org/wiki/Tic-tac-toe)                                                                |
| 2  |    [board_games/gomoku](https://github.com/opendilab/LightZero/tree/main/zoo/board_games/gomoku)    |                                ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                                |      ![original](https://github.com/opendilab/LightZero/tree/main/zoo/board_games/gomoku/gomoku.gif)       |                                                                  [env tutorial](https://en.wikipedia.org/wiki/Gomoku)                                                                  |
| 3  |  [board_games/connect4](https://github.com/opendilab/LightZero/tree/main/zoo/board_games/connect4)  |                                ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                                |        ![original](https://github.com/opendilab/LightZero/tree/main/zoo/board_games/connect4/connect4.gif)         |                                                                 [env tutorial](https://en.wikipedia.org/wiki/Connect4)                                                                 |
| 4  |             [game_2048](https://github.com/opendilab/LightZero/tree/main/zoo/game_2048)             |                                ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                                |         ![original](https://github.com/opendilab/LightZero/tree/main/zoo/game_2048/game_2048.gif)          |                                                                   [env tutorial](https://en.wikipedia.org/wiki/2048)                                                                   |
| 5  |           [chess](https://github.com/opendilab/LightZero/tree/main/zoo/board_games/chess)           |                                ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                                |       ![original](https://github.com/opendilab/LightZero/tree/main/zoo/board_games/chess/chess.gif)        |                                                                  [env tutorial](https://en.wikipedia.org/wiki/Chess)                                                                   |
| 6  |              [go](https://github.com/opendilab/LightZero/tree/main/zoo/board_games/go)              |                                ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                                |          ![original](https://github.com/opendilab/LightZero/tree/main/zoo/board_games/go/go.gif)           |                                                                    [env tutorial](https://en.wikipedia.org/wiki/Go)                                                                    |
| 7  |  [classic_control/cartpole](https://github.com/opendilab/LightZero/tree/main/zoo/classic_control)   |                                ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                                |                         ![original](./dizoo/classic_control/cartpole/cartpole.gif)                         |      [env tutorial](https://di-engine-docs.readthedocs.io/en/latest/13_envs/cartpole.html)<br>[环境指南](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/cartpole_zh.html)      |
| 8  |  [classic_control/pendulum](https://github.com/opendilab/LightZero/tree/main/zoo/classic_control)   |                                 ![continuous](https://img.shields.io/badge/-continous-green)                                  | ![original](https://github.com/opendilab/DI-engine/blob/main//dizoo/classic_control/pendulum/pendulum.gif) |      [env tutorial](https://di-engine-docs.readthedocs.io/en/latest/13_envs/pendulum.html)<br>[环境指南](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/pendulum_zh.html)      |
| 9  |           [box2d/lunarlander](https://github.com/opendilab/LightZero/tree/main/zoo/box2d)           | ![discrete](https://img.shields.io/badge/-discrete-brightgreen)  ![continuous](https://img.shields.io/badge/-continous-green) |   ![original](https://github.com/opendilab/DI-engine/blob/main//dizoo/box2d/lunarlander/lunarlander.gif)   |   [env tutorial](https://di-engine-docs.readthedocs.io/en/latest/13_envs/lunarlander.html)<br>[环境指南](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/lunarlander_zh.html)   |
| 10 |          [box2d/bipedalwalker](https://github.com/opendilab/LightZero/tree/main/zoo/box2d)          |                                 ![continuous](https://img.shields.io/badge/-continous-green)                                  |   ![original](https://github.com/opendilab/DI-engine/blob/main//dizoo/box2d/bipedalwalker/bipedalwalker.gif)    | [env tutorial](https://di-engine-docs.readthedocs.io/en/latest/13_envs/bipedalwalker.html)<br>[环境指南](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/bipedalwalker_zh.html) |
| 11 |                 [atari](https://github.com/opendilab/LightZero/tree/main/zoo/atari)                 |                                ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                                |            ![original](https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/atari.gif)             |         [env tutorial](https://di-engine-docs.readthedocs.io/en/latest/13_envs/atari.html)<br>[环境指南](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/atari_zh.html)         |
| 11 |                [mujoco](https://github.com/opendilab/LightZero/tree/main/zoo/mujoco)                |                                 ![continuous](https://img.shields.io/badge/-continous-green)                                  |           ![original](https://github.com/opendilab/DI-engine/blob/main/dizoo/mujoco/mujoco.gif)            |        [env tutorial](https://di-engine-docs.readthedocs.io/en/latest/13_envs/mujoco.html)<br>[环境指南](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/mujoco_zh.html)        |
| 12 |              [minigrid](https://github.com/opendilab/LightZero/tree/main/zoo/minigrid)              |                                ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                                |         ![original](https://github.com/opendilab/DI-engine/blob/main/dizoo/minigrid/minigrid.gif)          |      [env tutorial](https://di-engine-docs.readthedocs.io/en/latest/13_envs/minigrid.html)<br>[环境指南](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/minigrid_zh.html)      |

</details>

![discrete](https://img.shields.io/badge/-discrete-brightgreen) means discrete action space

![continuous](https://img.shields.io/badge/-continous-green) means continuous action space

- Some environments, like the LunarLander, support both types of action spaces. For continuous action space environments such as BipedalWalker and Pendulum, you can manually discretize them to obtain discrete action spaces. Please refer to [action_discretization_env_wrapper.py](https://github.com/opendilab/LightZero/blob/main/lzero/envs/wrappers/action_discretization_env_wrapper.py) for more details.

- This list is continually updated as we add more game environments to our collection.

# 动作空间
## 离散动作空间
1) action_space_size 
2) legal actions
这里的action_sape_size 就是确定的 比如只有四个动作, 向左, 向右, 向上, 向下.
同一个时刻是否只能采取某一个动作. 应该是确定的.

## 连续动作空间
action_space_size 
continuous_action_space = True
K = 20  # num_of_sampled_actions
连续 
1. 数值是连续, 同时采用多个动作.
2. 数值不连续, 同时采用多个动作.
3. 数据连续, 采用一个动作.
连续的一维, 二维, 三维动作空间.

当动作空间是连续的时，意味着智能体可以选择的动作是一个连续的范围，而不是离散的、有限的动作集合。在这样的情况下，动作可以取任意实数值或实数向量。

举例说明
自动驾驶汽车：

在自动驾驶领域，控制车速和方向的动作是连续的。假设智能体控制车辆的加速度和转向角度，那么动作空间就是一个二维的连续空间，其中：
加速度可以在一个连续的范围内变化，例如 ([-3, 3]) 米/秒²。
转向角度可以在一个连续的范围内变化，例如 ([-30^\circ, 30^\circ])。
机械臂控制：

在机器人学中，机械臂的动作通常也是连续的。例如，一个六自由度的机械臂，其每一个关节的旋转角度都可以在某个连续的范围内变化。这意味着动作空间是六维的，每个维度对应一个关节的角度。
游戏中的角色控制：

在某些视频游戏中，角色的移动控制也可以是连续的。例如，在一个飞行模拟器中，玩家可以控制飞机的俯仰（pitch）、偏航（yaw）和滚转（roll），每个控制都是一个连续值。这构成一个三维的连续动作空间。
数学描述
假设动作 ( a ) 是一个实数，那么连续动作空间可以表示为一个区间，例如 ( a \in [a_{\text{min}}, a_{\text{max}}] )。

如果动作 ( a ) 是一个向量，那么连续动作空间可以表示为一个向量的范围，例如 ( a \in \mathbb{R}^n )，其中每个分量 ( a_i ) 的取值范围是 ( [a_{i,\text{min}}, a_{i,\text{max}}] )。