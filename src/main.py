# 強化学習勉強用サンプルプログラム メイン処理

import argparse
from enum import Enum

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

# コマンドライン引数の解析
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='..\\config\\config.yaml',
        help='設定ファイル(YAML)のパス。')
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
    # 設定値の取得
    args = parse_args()
    config = read_config(args.config)

    # 環境の作成
    env_instance = ENV2CLS[args.env](config['environment'][args.env])

    # エージェントの作成
    agent_instance = ALGO2CLS[args.method](env_instance, config['agent'][args.method])

    # 選択したエージェントで学習+プレイ
    agent_instance.train()
    agent_instance.play()

if __name__ == '__main__':
    main()
