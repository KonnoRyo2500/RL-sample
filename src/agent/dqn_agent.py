# 強化学習勉強用サンプルプログラム DQNエージェントクラス

from agent.agent_base import AgentBase

# DQNエージェントクラス
class DqnAgent(AgentBase):
    # コンストラクタ
    def __init__(self, env, config):
        super().__init__(env, config)

    # 学習済みエージェントにエピソードをプレイさせる
    def play(self):
        pass

    # エージェントを学習させる
    def train(self):
        pass
