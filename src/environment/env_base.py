# 強化学習勉強用サンプルプログラム 環境ベースクラス

from abc import ABCMeta, abstractmethod

# 環境ベースクラス
# このクラスのインスタンスは作成できない(抽象クラス)
class EnvironmentBase(metaclass=ABCMeta):
    # コンストラクタ
    def __init__(self, config):
        self.config = config
        self.state = None

    # 環境全体における行動空間を取得
    @abstractmethod
    def get_whole_action_space(self):
        pass

    # 現状態における行動空間を取得
    @abstractmethod
    def get_current_action_space(self):
        pass

    # 指定された行動を実行し、報酬を得る
    @abstractmethod
    def exec_action(self, action):
        pass

    # 環境の状態空間を取得
    # 状態が多すぎて(もしくは無限に存在して)取得できない場合はNoneを返す
    @abstractmethod
    def get_state_space(self):
        pass

    # 現在の状態を取得
    @abstractmethod
    def get_state(self):
        pass

    # 現在の状態が終端状態かどうかを返す
    @abstractmethod
    def is_terminal_state(self):
        pass

    # 環境をリセットする
    @abstractmethod
    def reset(self):
        pass
