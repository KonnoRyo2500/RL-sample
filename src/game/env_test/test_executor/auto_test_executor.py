# 強化学習勉強用サンプルプログラム 自動環境テストクラス

from game.env_test.test_executor.test_executor_base import TestExecutorBase


# 自動環境テストクラス
class AutoTestExecutor(TestExecutorBase):
    # コンストラクタ
    def __init__(self, config):
        super().__init__(config)

    # テストを実施する
    def exec_test(self):
        print('自動テスト')
