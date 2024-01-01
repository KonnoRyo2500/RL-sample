# 強化学習勉強用サンプルプログラム メイン処理

import argparse

from environment.environment_factory import EnvironmentFactory
from agent.agent_factory import AgentFactory
from game.game_factory import GameFactory
from game.implementation.env_test.env_tester import EnvTester
from common.const_val import Agent, Environment


# コマンドライン引数の解析
def parse_args():
    parser = argparse.ArgumentParser()
    method_names = [str(method_name.value) for method_name in Agent]
    env_names = [str(env_name.value) for env_name in Environment]
    parser.add_argument(
        '--env',
        type=str,
        default=Environment.GridWorld.value,
        choices=env_names,
        help='利用する環境。')
    parser.add_argument(
        '--methods',
        type=str,
        default=[Agent.QLearning.value],
        choices=method_names,
        nargs='+',
        help='利用する強化学習アルゴリズム。')
    parser.add_argument(
        '--test-env',
        action='store_true',
        help='環境テストモードを有効にする。エージェントの学習は行われません。'
    )
    args = parser.parse_args()
    return args


# メイン処理
def main():
    # コマンドライン引数の取得
    args = parse_args()

    # 環境の作成
    env_instance = EnvironmentFactory.create_instance(args.env)

    # エージェントの作成
    if args.test_env:
        agent_instances = None
    else:
        agent_instances = [AgentFactory.create_instance(name, env_instance) for name in args.methods]

    # ゲームフレームワークの作成
    if args.test_env:
        # 環境テスト
        game_instance = EnvTester(env_instance, agent_instances)
    else:
        # 通常実行
        game_instance = GameFactory.create_instance(args.env, env_instance, agent_instances)

    # 指定された環境とエージェントで学習+プレイ
    game_instance.train_agent()
    game_instance.play()


if __name__ == '__main__':
    main()
