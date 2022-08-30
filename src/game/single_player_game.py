# 強化学習勉強用サンプルプログラム 一人用ゲームプレイ環境クラス

from game.game_base import GameBase


# 一人用ゲームプレイ環境クラス
class SinglePlayerGame(GameBase):
    # コンストラクタ
    def __init__(self):
        super().__init__()

    # エージェントを学習させる
    def train_agent(self):
        pass

    # 学習済みのエージェントでゲームをプレイする
    def play(self):
        pass
