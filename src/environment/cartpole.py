# 強化学習勉強用サンプルプログラム Cartpole環境クラス

from environment.env_base import EnvironmentBase

# Cartpole環境クラス
class Cartpole(EnvironmentBase):
    # コンストラクタ
    def __init__(self, config):
        super().__init__(config)

    # 環境全体における行動空間を取得
    def get_whole_action_space(self):
        pass

    # 現状態における行動空間を取得
    def get_current_action_space(self):
        pass

    # 指定された行動を実行し、報酬を得る
    def exec_action(self, action):
        pass

    # 環境の状態空間を取得
    # 状態が多すぎて(もしくは無限に存在して)取得できない場合はNoneを返す
    def get_state_space(self):
        pass

    # 現在の状態を取得
    def get_state(self):
        pass

    # 現在の状態が終端状態かどうかを返す
    def is_terminal_state(self):
        pass

    # 環境をリセットする
    def reset(self):
        pass
