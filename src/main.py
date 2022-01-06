# 強化学習勉強用サンプルプログラム メイン処理

from environment.grid_world import GridWorld
from agent.q_agent import QAgent

# メイン処理
def main():
    gw = GridWorld()
    q = QAgent(gw)

if __name__ == '__main__':
    main()
