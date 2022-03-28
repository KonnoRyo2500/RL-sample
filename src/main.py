# 強化学習勉強用サンプルプログラム メイン処理

import argparse
from enum import Enum

from common.config import read_config
import environment
import agent

# アルゴリズム名
class Algorithm(Enum):
    MonteCarlo = 'monte_carlo'
    QLearning = 'q_learning'
    Reinforce = 'reinforce'
    Sarsa = 'sarsa'
ALGORITHMS = [a.value for a in Algorithm]

# 環境名
class Environment(Enum):
    CartPole = 'cartpole'
    GridWorld = 'grid_world'
ENVS = [e.value for e in Environment]

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
        choices=ENVS,
        help='利用する環境。')
    parser.add_argument(
        '--method',
        type=str,
        default=Algorithm.QLearning.value,
        choices=ALGORITHMS,
        help='利用する強化学習アルゴリズム。')
    args = parser.parse_args()
    return args

# 環境の作成
def create_env(name, config):
    env_classes = [environment.Cartpole, environment.GridWorld]
    env_configs = [config[e] for e in ENVS]

    env_class = dict(zip(ENVS, env_classes))[name]
    env_config = dict(zip(ENVS, env_configs))[name]

    env_instance = env_class(env_config)

    return env_instance

# エージェントの作成
def create_agent(name, env, config):
    agent_classes = [agent.MonteCarloAgent, agent.QAgent, agent.ReinforceAgent, agent.SarsaAgent]
    agent_configs = [config[a] for a in ALGORITHMS]

    agent_class = dict(zip(ALGORITHMS, agent_classes))[name]
    agent_config = dict(zip(ALGORITHMS, agent_configs))[name]

    agent_instance = agent_class(env, agent_config)

    return agent_instance

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
