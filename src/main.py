# 強化学習勉強用サンプルプログラム メイン処理

import argparse

import environment
import agent
import game
from common.const_val import Agent, Environment
from game.single_player_game import SinglePlayerGame


# アルゴリズム名とエージェントクラスの対応
ALGO2CLS = {
    Agent.QLearning.value: agent.QAgent,
    Agent.Sarsa.value: agent.SarsaAgent,
    Agent.MonteCarlo.value: agent.MonteCarloAgent,
    Agent.Dqn.value: agent.DqnAgent,
}


# 環境名と環境クラスの対応
ENV2CLS = {
    Environment.GridWorld.value: environment.GridWorld,
    Environment.Cartpole.value: environment.Cartpole,
    Environment.Othello.value: environment.Othello,
}

# 環境名とゲームフレームワーククラスの対応
ENV2GAME = {
    Environment.GridWorld.value: game.SinglePlayerGame,
    Environment.Cartpole.value: game.SinglePlayerGame,
    Environment.Othello.value: game.OneVsOneGame,
}


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
    args = parser.parse_args()
    return args


# メイン処理
def main():
    # コマンドライン引数の取得
    args = parse_args()

    # 環境の作成
    env_instance = ENV2CLS[args.env]()
    # エージェントの作成
    agent_instances = [ALGO2CLS[name](env_instance) for name in args.methods]

    # ゲームフレームワークの作成
    game_instance = SinglePlayerGame(env_instance, agent_instances)

    # 指定された環境とエージェントで学習+プレイ
    game_instance.train_agent()
    game_instance.play()


if __name__ == '__main__':
    main()
