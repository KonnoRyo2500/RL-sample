# 強化学習勉強用サンプルプログラム Cartpole環境クラス
# 環境として、OpenAI GymのCartPole-v1を用いる

from enum import Enum
from itertools import product
import math

import gym

from environment.env_base import EnvironmentBase

# カートを押す方向
class PushDirection(Enum):
    Left = 0 # 左
    Right = 1 # 右

# 各状態の最小値、最大値
# カートの位置
CART_POSITION_MIN = -2.4
CART_POSITION_MAX = 2.4
# カートの速度
CART_VELOCITY_MIN = float('-inf')
CART_VELOCITY_MAX = float('inf')
# 棒の角度(度)
POLE_ANGLE_MIN = -24.8
POLE_ANGLE_MAX = 24.8
# 棒の速度
POLE_VELOCITY_MIN = float('-inf')
CART_VELOCITY_MAX = float('inf')

# Cartpole環境クラス
class Cartpole(EnvironmentBase):
    # コンストラクタ
    def __init__(self, config):
        super().__init__(config)

        self.env = gym.make('CartPole-v1')
        self.is_terminated = False
        self.state = self._convert_state(self.env.reset())

    # 環境全体における行動空間を取得
    def get_whole_action_space(self):
        return [dir for dir in PushDirection]

    # 現状態における行動空間を取得
    def get_current_action_space(self):
        # この環境では、行動空間は状態に依らない
        return self.get_whole_action_space()

    # 指定された行動を実行し、報酬を得る
    def exec_action(self, action):
        # 行動の実行
        next_state, reward, is_terminated, dbg_info = self.env.step(action.value)

        # 次状態の適用
        self.state = self._convert_state(next_state)
        self.is_terminated = is_terminated

        return reward

    # 環境の状態空間を取得
    # 状態が多すぎて(もしくは無限に存在して)取得できない場合はNoneを返す
    # Note: 各状態の分割数を増やしすぎると、状態空間のサイズが大きくなるため注意!
    def get_state_space(self):
        # 状態の分割数の範囲を取得
        rng_pos = range(self.config['position_part_num'])
        rng_cart_v = range(self.config['cart_velocity_part_num'])
        rng_angle = range(self.config['pole_angle_part_num'])
        rng_pole_v = range(self.config['pole_velocity_part_num'])

        # 状態空間の作成
        state_space = [(p, vc, a, vp) for p, vc, a, vp in product(rng_pos, rng_cart_v, rng_angle, rng_pole_v)]

        return state_space

    # 現在の状態を取得
    def get_state(self):
        return self.state

    # 現在の状態が終端状態かどうかを返す
    def is_terminal_state(self):
        return self.is_terminated

    # 環境をリセットする
    def reset(self):
        self.state = self._convert_state(self.env.reset())
        self.is_terminated = False

    # 状態を内部で扱いやすい形に変換する
    # list型に変換し、棒の角度をラジアンから度に変換する
    # さらに、状態を量子化する
    def _convert_state(self, original_state):
        rad2degree = lambda rad_angle: rad_angle * (180.0 / math.pi)
        state = list(original_state)
        state[2] = rad2degree(state[2])
        state = self._quantize_state(state)

        return state

    # 状態を量子化する
    # 状態が連続値なので、シンプルなアルゴリズムでもこの環境を利用できるようにする
    def _quantize_state(self, original_state):
        # 詳細な状態の取得
        cart_position, cart_velocity, pole_angle, pole_velocity = original_state

        # カート位置の量子化
        qs_pos = self._quantize_range(
            self.config['cart_position_min'],
            self.config['cart_position_max'],
            self.config['position_part_num'],
            cart_position)

        # カート速度の量子化
        qs_cart_v = self._quantize_range(
            self.config['cart_velocity_min'],
            self.config['cart_velocity_max'],
            self.config['cart_velocity_part_num'],
            cart_velocity)

        # 棒の角度の量子化
        qs_angle = self._quantize_range(
            self.config['pole_angle_min'],
            self.config['pole_angle_max'],
            self.config['pole_angle_part_num'],
            pole_angle)

        # 棒の速度の量子化
        qs_pole_v = self._quantize_range(
            self.config['pole_velocity_min'],
            self.config['pole_velocity_max'],
            self.config['pole_velocity_part_num'],
            pole_velocity)

        # 量子化された状態を得る
        quantized_state = (qs_pos, qs_cart_v, qs_angle, qs_pole_v)

        return quantized_state

    # 値の範囲を指定された個数に等分割し、連続値を量子化する
    # 注) min, maxは組み込み関数名で存在するため、あえて引数名は"minimum", "maximum"としている
    def _quantize_range(self, minimum, maximum, num_part, val):
        # [minimum, maximum)の範囲内でなければ、はみ出た側の区間に
        # 属しているものとみなす
        if val < minimum:
            return 0
        elif maximum <= val:
            return num_part - 1

        # 範囲の長さと区間の長さを取得する
        range_len = maximum - minimum
        section_len = range_len / num_part

        # この式で計算することで、forループ+if文を使わずとも
        # 区間のインデックスを求めることができる
        section_idx = int((val - minimum) // section_len)

        return section_idx
