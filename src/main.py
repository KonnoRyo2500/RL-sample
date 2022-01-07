# 強化学習勉強用サンプルプログラム メイン処理

from environment.grid_world import GridWorld
from agent.q_agent import QAgent
from agent.sarsa_agent import SarsaAgent

# メイン処理
def main():
    gw = GridWorld(wall_xlsx='..\\Grid.xlsx')
    q = QAgent(gw)
    q.train()
    q.play()

    gw.reset()
    s = SarsaAgent(gw)
    s.train()
    s.play()

if __name__ == '__main__':
    main()
