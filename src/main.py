# 強化学習勉強用サンプルプログラム メイン処理

import argparse
from enum import Enum
import os.path as op

from common.config import read_config
import environment
import agent

# アルゴリズム名
class Algorithm(Enum):
    QLearning = 'q_learning' # Q学習
    Sarsa = 'sarsa' # SARSA
    MonteCarlo = 'monte_carlo' # モンテカルロ法
    DQN = 'dqn' # DQN

# 環境名
class Environment(Enum):
    GridWorld = 'grid_world' # Grid World
    Cartpole = 'cartpole' # Cartpole

# アルゴリズム名とエージェントクラスの対応
ALGO2CLS = {
    Algorithm.QLearning.value: agent.QAgent,
    Algorithm.Sarsa.value: agent.SarsaAgent,
    Algorithm.MonteCarlo.value: agent.MonteCarloAgent,
    Algorithm.DQN.value: agent.DqnAgent,
}

# 環境名と環境クラスの対応
ENV2CLS = {
    Environment.GridWorld.value: environment.GridWorld,
    Environment.Cartpole.value: environment.Cartpole,
}

# 設定ファイル名
CONFIG_NAME = 'config.yaml'

# コマンドライン引数の解析
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env',
        type=str,
        default=Environment.GridWorld.value,
        choices=list(ENV2CLS.keys()),
        help='利用する環境。')
    parser.add_argument(
        '--method',
        type=str,
        default=Algorithm.QLearning.value,
        choices=list(ALGO2CLS.keys()),
        help='利用する強化学習アルゴリズム。')
    args = parser.parse_args()
    return args

# メイン処理
def main():
    # コマンドライン引数の取得
    args = parse_args()

    # 設定ファイルの読み込み
    config_root = op.join(op.dirname(__file__), '..', 'config') # リポジトリルート直下、"config"フォルダ
    env_config_path = op.join(config_root, 'environment', args.env, CONFIG_NAME)
    agent_config_path = op.join(config_root, 'agent', args.method, CONFIG_NAME)

    env_config = read_config(env_config_path)
    agent_config = read_config(agent_config_path)

    # 環境の作成
    env_instance = ENV2CLS[args.env](env_config)

    # エージェントの作成
    agent_instance = ALGO2CLS[args.method](env_instance, agent_config)

    # 選択したエージェントで学習+プレイ
    agent_instance.train()
    agent_instance.play()

if __name__ == '__main__':
    main()
