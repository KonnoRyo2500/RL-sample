# 強化学習勉強用サンプルプログラム ゲームフレームワークファクトリークラス

from common.const_val import Environment
from game.implementation.single_player.single_player_game import SinglePlayerGame
from game.implementation.multi_player.multi_player_game import MultiPlayerGame


# ゲームフレームワークファクトリークラス
class GameFactory:
    # 与えられた環境名、環境、エージェントから、ゲームフレームワークのインスタンスを作成する
    @staticmethod
    def create_instance(env_name, env, agents):
        if env_name == Environment.GridWorld.value:
            return SinglePlayerGame(env, agents)
        elif env_name == Environment.Cartpole.value:
            return SinglePlayerGame(env, agents)
        elif env_name == Environment.Othello.value:
            return MultiPlayerGame(env, agents)
        else:
            return None
