# 強化学習勉強用サンプルプログラム 複数人プレイヤーゲームフレームワーククラス

from game.base.game_base import GameBase
from common.const_val import Game


# 複数人プレイヤーゲームフレームワーククラス
class MultiPlayerGame(GameBase):
    # コンストラクタ
    def __init__(self, env, agents):
        super().__init__(env, agents, Game.MultiPlayer.value)

        self.player = 0  # 現手番でのプレイヤー
        self.env.change_player(self.player)

    # エージェントを学習させる
    def train_agent(self):
        # すべてのプレイヤーが終端状態になるまで、ゲームを続ける
        is_terminated = [False] * len(self.agents)
        while not all(is_terminated):
            # 現在のプレイヤーがすでに終端状態にある
            if self.env.is_terminated():
                is_terminated[self.player] = True
                self._switch_to_next_player()
                continue

            # エージェントが次の行動を決定する
            agent = self.agents[self.player]
            action = agent.decide_action(self.env)

            # 決定した行動を環境に対して行い、報酬を得る
            reward = self.env.exec_action(action)

            # エージェントは、得られた報酬と変化した環境の状態をもとに自身のパラメータを更新する
            agent.feedback(reward, self.env)

            # 次のプレイヤーに交代
            self._switch_to_next_player()

    # 学習済みのエージェントでゲームをプレイする
    def play(self):
        # すべてのプレイヤーが終端状態になるまで、ゲームを続ける
        is_terminated = [False] * len(self.agents)
        while not all(is_terminated):
            # 現在のプレイヤーがすでに終端状態にある
            if self.env.is_terminated():
                is_terminated[self.player] = True
                self._switch_to_next_player()
                continue

            # エージェントが次の行動を決定する
            agent = self.agents[self.player]
            action = agent.decide_action(self.env)

            # 決定した行動を環境に対して行い、報酬を得る
            reward = self.env.exec_action(action)

            # 行動を表示する
            print(f'行動{action}を選択し、報酬{reward}が得られました')

            # 次のプレイヤーに交代
            self._switch_to_next_player()

    # 次のプレイヤーに交代する
    def _switch_to_next_player(self):
        self.player = (self.player + 1) % len(self.agents)
        self.env.change_player(self.player)
