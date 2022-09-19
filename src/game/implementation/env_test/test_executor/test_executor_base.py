# 強化学習勉強用サンプルプログラム 環境テスト実施ベースクラス

from abc import ABCMeta, abstractmethod


# 環境テスト実施ベースクラス
# 環境のテストを実際に行うためのクラスのベースクラス
class TestExecutorBase(metaclass=ABCMeta):
    # コンストラクタ
    def __init__(self, config, env):
        self.config = config
        self.env = env

    # テストを実施する
    @abstractmethod
    def exec_test(self):
        pass
