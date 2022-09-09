# 強化学習勉強用サンプルプログラム 1vs1対戦ゲームフレームワーククラス

from game.game_base import GameBase


# 1vs1対戦ゲームフレームワーククラス
class OneVsOneGame(GameBase):
    # コンストラクタ
    def __init__(self, env, agents, env_config, agent_configs):
        super().__init__(env, agents, env_config, agent_configs)

    # エージェントを学習させる
    def train_agent(self):
        pass

    # 学習済みのエージェントでゲームをプレイする
    def play(self):
        pass
