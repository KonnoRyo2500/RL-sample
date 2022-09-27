# 強化学習勉強用サンプルプログラム 環境テストコマンド実行クラス

# 環境テストコマンド実行クラス
class CommandExecutor:
    # コンストラクタ
    def __init__(self, env):
        self.env = env

    # コマンドの実行
    def exec_command(self, cmd):
        name = cmd[0]  # コマンド名
        args = cmd[1:] if len(cmd) >= 2 else []  # 引数

        # コマンド名に応じた操作を環境に対して実行する
        func_from_name = {
            'print': self._exec_print,
            'action': self._exec_action,
            'reset': self._exec_reset,
            'next-player': self._exec_next_player,
        }

        print(f'実行開始: {cmd}')
        if name not in func_from_name.keys():
            print('有効なコマンド名は以下の通りです。コマンドは実行されませんでした。')
            print(func_from_name.keys())
            return

        exec_func = func_from_name[name]
        exec_func(args)
        print('実行終了')

    # 'print' コマンドの実行
    def _exec_print(self, args):
        # 引数チェック(個数)
        if len(args) == 0:
            print("'print' コマンドには引数が1個必要です。コマンドは実行されませんでした")
            return
        elif len(args) >= 2:
            print("'print' コマンドに必要な引数は1個のみです。2番目以降の引数は無視されます")

        # 表示対象名と表示対象
        target_from_name = {
            'state': self.env.get_state(),
            'state-space': self.env.get_state_space(),
            'available-action': self.env.get_available_actions(),
            'action-space': self.env.get_action_space(),
            'is-terminal-state': self.env.is_terminal_state(),
            'player': self.env.get_player()
        }

        # 引数チェック(表示対象名として有効かどうか)
        target_name = args[0]
        if target_name not in target_from_name.keys():
            print("'print' コマンドの引数として指定できる文字列は以下の通りです。コマンドは実行されませんでした")
            print(target_from_name.keys())
            return

        # 指定された表示対象を表示
        target = target_from_name[target_name]
        print(f"表示: {target}")

    # 'action' コマンドの実行
    def _exec_action(self, args):
        # 引数チェック(個数)
        if len(args) == 0:
            print("'action' コマンドには引数が1個必要です。コマンドは実行されませんでした")
            return
        elif len(args) >= 2:
            print("'action' コマンドに必要な引数は1個のみです。2番目以降の引数は無視されます")

        # 引数チェック(インデックスとして有効かどうか)
        if not args[0].isdigit():
            print("'action' コマンドの引数は正の10進整数で指定してください。コマンドは実行されませんでした")
            return

        # 環境に依存しない実装にするため、行動を直接指定するのではなく、インデックスで指定する
        idx = int(args[0])
        available_action = self.env.get_available_actions()

        # 指定した行動を環境に対して実行する
        if idx >= len(available_action):
            print(f"'action' コマンドで指定したインデックスが範囲外です。選択可能行動数: {len(available_action)}")
        else:
            action = available_action[idx]
            reward = self.env.exec_action(action)
            print(f'行動 {action} が実行され、報酬 {reward} が得られました')

    # 'reset' コマンドの実行
    def _exec_reset(self, args):
        # 引数チェック(個数)
        if len(args) > 0:
            print("'reset' コマンドに引数は不要です。指定された引数は無視されます")

        self.env.reset()

    # 'next-player'コマンドの実行
    def _exec_next_player(self, args):
        # 引数チェック(個数)
        if len(args) > 0:
            print("'next-player' コマンドに引数は不要です。指定された引数は無視されます")

        # 環境に対して、プレイヤーの交代を指示する
        self.env.switch_to_next_player()
