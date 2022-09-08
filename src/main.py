# 強化学習勉強用サンプルプログラム メイン処理

import argparse
from enum import Enum
import os.path as op

import yaml

import environment
import agent
import game
from common.dirs import AGENT_CONFIG_DIR, ENV_CONFIG_DIR
from game.single_player_game import SinglePlayerGame


# アルゴリズム名
class Algorithm(Enum):
    QLearning = 'q_learning'  # Q学習
    Sarsa = 'sarsa'  # SARSA
    MonteCarlo = 'monte_carlo'  # モンテカルロ法
    DQN = 'dqn'  # DQN


# 環境名
class Environment(Enum):
    GridWorld = 'grid_world'  # Grid World
    Cartpole = 'cartpole'  # Cartpole
    Othello = 'othello'  # オセロ


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
    Environment.Othello.value: environment.Othello,
}

# 環境名とプレイ環境クラスの対応
ENV2GAME = {
    Environment.GridWorld.value: game.SinglePlayerGame,
    Environment.Cartpole.value: game.SinglePlayerGame,
    Environment.Othello.value: game.OneVsOneGame,
}


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
        '--methods',
        type=str,
        default=[Algorithm.QLearning.value],
        choices=list(ALGO2CLS.keys()),
        nargs='+',
        help='利用する強化学習アルゴリズム。')
    args = parser.parse_args()
    return args


# 設定ファイル(YAML)を読み込む
# 設定ファイルのパスは"<directory>\<prefix>_config.yaml"
def load_config(directory, prefix):
    # 設定ファイルパスの組み立て
    suffix = "_config"
    ext = ".yaml"
    name = prefix + suffix + ext
    path = op.join(directory, name)

    if not op.exists(path):
        raise FileNotFoundError(f'設定ファイル {path} が存在しません。')

    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


# メイン処理
def main():
    # コマンドライン引数の取得
    args = parse_args()

    # 設定ファイル読み込み
    env_config = load_config(ENV_CONFIG_DIR(args.env), args.env)
    agent_configs = [load_config(AGENT_CONFIG_DIR(name), name) for name in args.methods]

    # 環境の作成
    env_instance = ENV2CLS[args.env](env_config)
    # エージェントの作成
    agent_instances = [ALGO2CLS[name](env_instance, config) for name, config in zip(args.methods, agent_configs)]

    # プレイ環境の作成
    game_instance = SinglePlayerGame(env_instance, agent_instances, env_config, agent_configs)

    # 指定された環境とエージェントで学習+プレイ
    game_instance.train_agent()
    game_instance.play()


if __name__ == '__main__':
    main()
