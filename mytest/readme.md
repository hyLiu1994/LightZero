# Gym 环境学习

## CartPole
1) 机器人强化学习之使用 OpenAI Gym [教程与笔记](https://zhuanlan.zhihu.com/p/40673328)
2) CartPole[倒立摆](https://zhuanlan.zhihu.com/p/570695189)

## 课程
1) Gym [课程笔记](https://exp-blog.com/ai/gym-bi-ji-01-huan-jing-da-jian-yu-ji-ben-gai-nian/)

# MCTS 
## 入门
1. 蒙特卡洛树搜索 [MCTS 入门](https://zhuanlan.zhihu.com/p/26335999)


# error
## mujoco
1. Can't get mujoco example working on [colab](https://github.com/opendilab/LightZero/issues/56)
2. Mujoco&Mujoco-py安装教程以及常见[报错解决方法](https://zhuanlan.zhihu.com/p/352304615)
3.  mujoco-py does not support versions of MuJoCo [after 2.1.0.](https://github.com/openai/mujoco-py)
4. 安装mujoco_py并测试时[提示](https://blog.csdn.net/m0_38122847/article/details/133781095)
pip uninstall cython
pip install cython==0.29.21 
5. No such file or directory: 'patchelf' on mujoco-py [installation](https://github.com/openai/mujoco-py/issues/652)
6. AttributeError: 'MujocoEnv' object has no attribute '[_observation_space](https://github.com/opendilab/DI-engine/issues/473)'
7. AttributeError: module 'mujoco_py' has no attribute ['utils'](https://github.com/openai/mujoco-py/issues/464)
8. 安装mujoco_py遇到的一些[问题与解决方法](https://blog.csdn.net/weixin_44420419/article/details/116231500)
9. [api_key ](https://www.roboti.us/license.html)

# 随机性分析

## MCTS
1. c/c++ 版本的随机性使用随机种子控制不住, 但是python 版本的可以控制.
2. collect env 以及 eval env 初始化的obs不同的