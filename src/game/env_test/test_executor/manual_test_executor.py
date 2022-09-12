# 強化学習勉強用サンプルプログラム 手動環境テストクラス

from game.env_test.test_executor.test_executor_base import TestExecutorBase
from game.env_test.test_executor.command_executor import CommandExecutor


# 手動環境テストクラス
class ManualTestExecutor(TestExecutorBase):
    # コンストラクタ
    def __init__(self, config, env):
        super().__init__(config, env)

    # テストを実施する
    def exec_test(self):
        print('手動テスト開始')
        cmd_executor = CommandExecutor(self.env)
        terminate_cmd = 'quit'  # 終了コマンド

        # 終了コマンドが入力されるまで、ユーザからのコマンド入力を受け付ける
        while True:
            original_cmd = input(f"コマンドを入力してください('{terminate_cmd}' で終了)\n> ")

            cmd = original_cmd.strip().split(' ')
            if cmd[0] == terminate_cmd:
                break

            cmd_executor.exec_command(cmd)
