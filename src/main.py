# 強化学習勉強用サンプルプログラム メイン処理

import argparse

from common.config import read_config
from environment.grid_world import GridWorld
from agent.q_agent import QAgent
from agent.sarsa_agent import SarsaAgent

# コマンドライン引数の解析
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='..\\config\\config.yaml',
        help='設定ファイル(YAML)のパス。')
    args = parser.parse_args()
    return args

# メイン処理
def main():
    # 設定値の取得
    args = parse_args()
    config = read_config(args.config)
    grid_world_config = config['environment']['grid_world']
    q_config = config['agent']['q_learning']
    sarsa_config = config['agent']['sarsa']

    # 環境の作成
    gw = GridWorld(config=grid_world_config)

    # Q学習で学習+プレイ
    q = QAgent(gw, q_config)
    q.train()
    q.play()

    # SARSAで学習+プレイ
    gw.reset()
    s = SarsaAgent(gw, sarsa_config)
    s.train()
    s.play()

if __name__ == '__main__':
    main()
