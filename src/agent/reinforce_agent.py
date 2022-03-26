# 強化学習勉強用サンプルプログラム REINFORCEエージェントクラス

from agent.agent_base import AgentBase

# REINFORCEエージェントクラス
class ReinforceAgent(AgentBase):
    # コンストラクタ
    def __init__(self, env, config):
        super().__init__(env, config)

    # 学習済みエージェントにエピソードをプレイさせる
    def play(self):
        pass

    # エージェントを学習させる
    def train(self):
        pass
