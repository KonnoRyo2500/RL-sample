# 強化学習勉強用サンプルプログラム モンテカルロ法エージェントクラス

from agent.agent_base import AgentBase

# モンテカルロ法エージェントクラス
class MonteCarloAgent(AgentBase):
    # コンストラクタ
    def __init__(self, env, config):
        super().__init__(env, config)

    # 学習した価値関数を基にエピソードをプレイ
    def play(self):
        pass

    # 学習の実行
    def train(self):
        pass
