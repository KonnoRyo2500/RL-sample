# 強化学習勉強用サンプルプログラム 環境ファクトリークラス

from common.const_val import Environment
from environment.implementation.grid_world.grid_world import GridWorld
from environment.implementation.cartpole.cartpole import Cartpole
from environment.implementation.othello.othello import Othello


# 環境ファクトリークラス
class EnvironmentFactory:
    # 与えられた環境名から、環境のインスタンスを作成する
    @staticmethod
    def create_instance(name):
        if name == Environment.GridWorld.value:
            return GridWorld()
        elif name == Environment.Cartpole.value:
            return Cartpole()
        elif name == Environment.Othello.value:
            return Othello()
        else:
            return None
