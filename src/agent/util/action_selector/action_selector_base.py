# 強化学習勉強用サンプルプログラム 行動選択ベースクラス

from abc import ABCMeta, abstractmethod

# 行動選択ベースクラス
# このクラスのインスタンスは作成できない(抽象クラス)
class ActionSelectorBase(metaclass=ABCMeta):
    # 具体的な行動選択アルゴリズムはサブクラスで実装すること。

    # コンストラクタ
    def __init__(self, action_space):
        self.action_space = action_space # 環境の行動空間

    # 行動価値に基づき、行動を選択する
    @abstractmethod
    def select_action(self, state, q_values):
        pass
