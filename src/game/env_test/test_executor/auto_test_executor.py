# 強化学習勉強用サンプルプログラム 自動環境テストクラス

import os.path as op

from common.dirs import GAME_CONFIG_DIR
from common.const_val import Game
from game.env_test.test_executor.test_executor_base import TestExecutorBase
from game.env_test.test_executor.command_executor import CommandExecutor


# 自動環境テストクラス
class AutoTestExecutor(TestExecutorBase):
    # コンストラクタ
    def __init__(self, config, env):
        super().__init__(config, env)

    # テストを実施する
    def exec_test(self):
        print('自動テスト開始')
        # シナリオの読み込み
        scenario = self._load_scenario()

        # コマンドの実行
        cmd_executor = CommandExecutor(self.env)
        for cmd in scenario:
            cmd_executor.exec_command(cmd)

    # シナリオファイルを読み込む
    def _load_scenario(self):
        # シナリオファイルは、環境テストフレームワークの設定ファイルと同じディレクトリに配置する
        path = op.join(GAME_CONFIG_DIR(Game.EnvTester.value), self.config['scenario_file'])
        if not op.exists(path):
            raise RuntimeError(f'シナリオファイル {path} が存在しません')

        with open(path, 'r') as f:
            scenario = [cmd.strip().split(' ') for cmd in f.readlines()]

        return scenario
