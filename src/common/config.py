# 強化学習勉強用サンプルプログラム 設定取得処理

import os.path as op

import yaml

# 設定ファイル(YAML)を読み込む
# 設定値のデータ構造はpyyamlの定義に従う
def read_config(path):
    if not op.exists(path):
        raise FileNotFoundError(f'設定ファイル {path} が存在しません。')

    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config
