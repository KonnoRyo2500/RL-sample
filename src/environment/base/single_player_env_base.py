# 強化学習勉強用サンプルプログラム 1人ゲーム用環境ベースクラス

from abc import ABC

from environment.base.env_base import EnvironmentBase


# 1人ゲーム用環境ベースクラス
# このクラスのインスタンスは作成できない(抽象クラス)
class SinglePlayerEnvironmentBase(ABC, EnvironmentBase):
    # 環境ベースクラスから追加で実装する関数や変数はなし
    pass
