# 強化学習勉強用サンプルプログラム 環境ベースクラス

from abc import ABCMeta, abstractmethod

from common.dirs import ENV_CONFIG_DIR
from common.config import load_config


# 環境ベースクラス
# このクラスのインスタンスは作成できない(抽象クラス)
class EnvironmentBase(metaclass=ABCMeta):
    # コンストラクタ
    def __init__(self, name):
        self.config = self._load_config(name)
        self.state = None

    # 環境の行動空間を取得
    @abstractmethod
    def get_action_space(self):
        pass

    # 現在選択可能な行動を取得
    @abstractmethod
    def get_available_actions(self):
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

    # 設定ファイルを読み込む
    def _load_config(self, name):
        dir_name = ENV_CONFIG_DIR(name)
        file_name = f'{name}_config.yaml'

        return load_config(dir_name, file_name)
