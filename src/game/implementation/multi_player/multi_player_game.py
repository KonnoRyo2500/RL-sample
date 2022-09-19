# 強化学習勉強用サンプルプログラム 複数人プレイヤーゲームフレームワーククラス

from game.base.game_base import GameBase
from common.const_val import Game


# 複数人プレイヤーゲームフレームワーククラス
class MultiPlayerGame(GameBase):
    # コンストラクタ
    def __init__(self, env, agents):
        super().__init__(env, agents, Game.MultiPlayer.value)

    # エージェントを学習させる
    def train_agent(self):
        pass

    # 学習済みのエージェントでゲームをプレイする
    def play(self):
        pass
