# 強化学習勉強用サンプルプログラム 定数定義

from enum import Enum

# 注意:
#   1) ここには、ユーザによらない定数を記述すること。
#      ユーザが設定しうる値は、設定ファイルに記載する。
#   2) 複数ファイルで共通して利用する定数のみ定義すること。
#      1つのファイルでしか使わない定数は、各ファイルに定義する。


# エージェント名
class Agent(Enum):
    QLearning = 'q_learning'  # Q学習
    Sarsa = 'sarsa'  # SARSA
    MonteCarlo = 'monte_carlo'  # モンテカルロ法
    Dqn = 'dqn'  # DQN


# 環境名
class Environment(Enum):
    GridWorld = 'grid_world'  # Grid World
    Cartpole = 'cartpole'  # Cartpole
    Othello = 'othello'  # オセロ


# ゲームフレームワーク名
class Game(Enum):
    OneVsOne = '1vs1'  # 1vs1対戦ゲーム
    SinglePlayer = 'single_player'  # 一人用ゲーム
