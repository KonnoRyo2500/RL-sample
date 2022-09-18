# 強化学習勉強用サンプルプログラム 1vs1対戦ゲームフレームワーククラス

from game.base.game_base import GameBase
from common.const_val import Game


# 1vs1対戦ゲームフレームワーククラス
class OneVsOneGame(GameBase):
    # コンストラクタ
    def __init__(self, env, agents):
        super().__init__(env, agents, Game.OneVsOne.value)

    # エージェントを学習させる
    def train_agent(self):
        pass

    # 学習済みのエージェントでゲームをプレイする
    def play(self):
        pass
