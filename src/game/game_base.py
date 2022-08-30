# 強化学習勉強用サンプルプログラム ゲームプレイ環境ベースクラス

from abc import ABCMeta, abstractmethod


# ゲームプレイ環境ベースクラス
# このクラスのインスタンスは作成できない(抽象クラス)
class GameBase(metaclass=ABCMeta):
    # コンストラクタ
    def __init__(self):
        pass

    # エージェントを学習させる
    @abstractmethod
    def train_agent(self):
        pass

    # 学習済みのエージェントでゲームをプレイする
    @abstractmethod
    def play(self):
        pass
