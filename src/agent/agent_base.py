# 強化学習勉強用サンプルプログラム エージェントベースクラス

from abc import ABCMeta, abstractmethod
from enum import Enum, auto


# エージェントベースクラス
# このクラスのインスタンスは作成できない(抽象クラス)
class AgentBase(metaclass=ABCMeta):
    # 具体的な強化学習アルゴリズムはサブクラスで実装すること。

    # 動作モード
    class OperationMode(Enum):
        Train = auto()
        Play = auto()

    # コンストラクタ
    def __init__(self, env, config):
        self.config = config
        self.mode = AgentBase.OperationMode.Train

    # 環境の情報を参照し、次の行動を決定する
    @abstractmethod
    def decide_action(self, env):
        pass

    # 環境からの情報を自身にフィードバックする
    @abstractmethod
    def feedback(self, reward, env):
        pass

    # エージェントを学習モードにする
    def switch_to_train_mode(self):
        self.mode = AgentBase.OperationMode.Train

    # エージェントをプレイモードにする
    def switch_to_play_mode(self):
        self.mode = AgentBase.OperationMode.Play
