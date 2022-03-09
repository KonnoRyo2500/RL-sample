# 強化学習勉強用サンプルプログラム メイン処理

import argparse

from common.config import read_config
from environment.grid_world import GridWorld
from environment.cartpole import Cartpole
from agent.q_agent import QAgent
from agent.sarsa_agent import SarsaAgent
from agent.monte_carlo_agent import MonteCarloAgent

# アルゴリズム名
METHODS = ['q_learning', 'sarsa', 'monte_carlo']
# 環境名
ENVS = ['grid_world', 'cartpole']

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
        default='grid_world',
        choices=ENVS,
        help='利用する環境。')
    parser.add_argument(
        '--method',
        type=str,
        default='q_learning',
        choices=METHODS,
        help='利用する強化学習アルゴリズム。')
    args = parser.parse_args()
    return args

# 環境の作成
def create_env(name, config):
    env_classes = [GridWorld, Cartpole]
    env_configs = [config['grid_world'], config['cartpole']]

    env_class = dict(zip(ENVS, env_classes))[name]
    env_config = dict(zip(ENVS, env_configs))[name]

    env = env_class(env_config)

    return env

# エージェントの作成
def create_agent(method, env, config):
    agent_classes = [QAgent, SarsaAgent, MonteCarloAgent]
    agent_configs = [config['q_learning'], config['sarsa'], config['monte_carlo']]

    agent_class = dict(zip(METHODS, agent_classes))[method]
    agent_config = dict(zip(METHODS, agent_configs))[method]

    agent = agent_class(env, agent_config)

    return agent

# メイン処理
def main():
    # 設定値の取得
    args = parse_args()
    config = read_config(args.config)

    # 環境の作成
    env = create_env(args.env, config['environment'])

    # エージェントの作成
    agent = create_agent(args.method, env, config['agent'])

    # 選択したエージェントで学習+プレイ
    agent.train()
    agent.play()

if __name__ == '__main__':
    main()
