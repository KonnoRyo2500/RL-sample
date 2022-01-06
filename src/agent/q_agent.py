# 強化学習勉強用サンプルプログラム Q学習エージェントクラス

from agent.agent_base import AgentBase

# Q学習エージェントクラス
class QAgent(AgentBase):
    # コンストラクタ
    def __init__(self, env):
        super().__init__(env)

    # 学習した価値関数を基にエピソードをプレイ
    def play(self):
        pass

    # 学習の実行
    def train(self):
        pass

    # エピソードの実行
    def episode(self):
        pass

    # 1ステップ実行
    def step(self):
        pass

