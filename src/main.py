# 強化学習勉強用サンプルプログラム メイン処理

import argparse

from environment.grid_world import GridWorld
from agent.q_agent import QAgent
from agent.sarsa_agent import SarsaAgent

# コマンドライン引数の解析
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--grid_xlsx',
        type=str,
        default='..\\Grid.xlsx',
        help='盤面情報を定義するExcelシートへのパス。')
    args = parser.parse_args()
    return args

# メイン処理
def main():
    args = parse_args()

    gw = GridWorld(wall_xlsx=args.grid_xlsx)
    q = QAgent(gw)
    q.train()
    q.play()

    gw.reset()
    s = SarsaAgent(gw)
    s.train()
    s.play()

if __name__ == '__main__':
    main()
