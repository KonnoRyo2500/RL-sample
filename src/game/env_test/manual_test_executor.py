# 強化学習勉強用サンプルプログラム 手動環境テストクラス

from game.env_test.test_executor_base import TestExecutorBase


# 手動環境テストクラス
class ManualTestExecutor(TestExecutorBase):
    # コンストラクタ
    def __init__(self, config):
        super().__init__(config)

    # テストを実施する
    def exec_test(self):
        print('手動テスト')
