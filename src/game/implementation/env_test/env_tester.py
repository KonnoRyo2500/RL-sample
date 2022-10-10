# 強化学習勉強用サンプルプログラム 環境テスト用フレームワーククラス

from game.base.game_base import GameBase
from game.implementation.env_test.test_executor.auto_test_executor import AutoTestExecutor
from game.implementation.env_test.test_executor.manual_test_executor import ManualTestExecutor
from common.const_val import Game


# 環境テスト用フレームワーククラス
class EnvTester(GameBase):
    # コンストラクタ
    def __init__(self, env, agents):
        super().__init__(env, agents, Game.EnvTester.value)

    # エージェントを学習する
    def train_agent(self):
        # Do Nothing
        pass

    # 学習済みのエージェントでゲームをプレイする
    # 本フレームワークでは、この関数で環境のテストを行うものとする
    def play(self):
        # テスト実施インスタンスの作成
        executor_classes = {
            'auto': AutoTestExecutor,  # 自動テスト
            'manual': ManualTestExecutor,  # 手動テスト
        }

        if self.config.test_type not in executor_classes.keys():
            raise RuntimeError("テスト種別には 'auto' もしくは 'manual' のいずれかを指定してください")

        executor_class = executor_classes[self.config.test_type]
        executor_instance = executor_class(self.config, self.env)

        # テスト実施
        executor_instance.exec_test()
