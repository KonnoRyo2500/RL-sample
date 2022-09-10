# 強化学習勉強用サンプルプログラム ディレクトリ定義

import os.path as op

# リポジトリルートディレクトリ
REPO_ROOT = op.join(op.dirname(__file__), '..', '..')

# 設定ファイルディレクトリ
CONFIG_DIR = op.join(REPO_ROOT, 'config')

# エージェント用設定ファイルディレクトリ
AGENT_CONFIG_DIR = lambda agent_name: op.join(CONFIG_DIR, 'agent', agent_name)
# 環境用設定ファイルディレクトリ
ENV_CONFIG_DIR = lambda env_name: op.join(CONFIG_DIR, 'environment', env_name)
# ゲームフレームワーク設定ファイルディレクトリ
GAME_CONFIG_DIR = lambda game_name: op.join(CONFIG_DIR, 'game', game_name)
