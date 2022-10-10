# 強化学習勉強用サンプルプログラム 設定ファイル操作用関数群

import os.path as op
from collections import namedtuple

import yaml


# 設定ファイルを読み込む
def load_config(dir_name, file_name):
    # 設定ファイルパスの組み立て
    path = op.join(dir_name, file_name)

    if not op.exists(path):
        raise FileNotFoundError(f'設定ファイル {path} が存在しません。')

    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config_class = namedtuple('Config', config.keys())
    config = config_class(**config)

    return config
