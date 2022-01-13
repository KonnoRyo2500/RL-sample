# 強化学習勉強用サンプルプログラム メイン処理

import argparse

from common.config import read_config
from environment.grid_world import GridWorld
from agent.q_agent import QAgent
from agent.sarsa_agent import SarsaAgent
from agent.monte_carlo_agent import MonteCarloAgent

# コマンドライン引数の解析
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='..\\config\\config.yaml',
        help='設定ファイル(YAML)のパス。')
    parser.add_argument(
        '--method',
        type=str,
        default='q_learning',
        choices=['q_learning', 'sarsa', 'monte_carlo'],
        help='利用する強化学習アルゴリズム。')
    args = parser.parse_args()
    return args

# エージェントの作成
def create_agent(method, env, config):
    agent_class = {
        'q_learning': QAgent,
        'sarsa': SarsaAgent,
        'monte_carlo': MonteCarloAgent,
    }[method]
    agent_config = {
        'q_learning': config['q_learning'],
        'sarsa': config['sarsa'],
        'monte_carlo': config['monte_carlo'],
    }[method]

    agent = agent_class(env, agent_config)

    return agent

# メイン処理
def main():
    # 設定値の取得
    args = parse_args()
    config = read_config(args.config)

    # 環境の作成
    gw = GridWorld(config=config['environment']['grid_world'])

    # エージェントの作成
    agent = create_agent(args.method, gw, config['agent'])

    # 選択したエージェントで学習+プレイ
    agent.train()
    agent.play()

if __name__ == '__main__':
    main()
