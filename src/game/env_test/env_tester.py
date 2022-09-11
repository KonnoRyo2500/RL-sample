# 強化学習勉強用サンプルプログラム 環境テスト用フレームワーククラス

from game.game_base import GameBase
from common.const_val import Game


# 環境テスト用フレームワーククラス
class EnvTester(GameBase):
    # コンストラクタ
    def __init__(self, env, agents):
        super().__init__(env, agents, Game.EnvTester.value)

    # エージェントを学習する
    def train_agent(self):
        # Do Nothing
        pass

    # 学習済みのエージェントでゲームをプレイする
    # 本フレームワークでは、この関数で環境のテストを行うものとする
    def play(self):
        pass