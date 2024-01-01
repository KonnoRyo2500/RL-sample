# 強化学習勉強用サンプルプログラム エージェントファクトリークラス

from common.const_val import Agent
from agent.implementation.q_learning.q_agent import QAgent
from agent.implementation.sarsa.sarsa_agent import SarsaAgent
from agent.implementation.monte_carlo.monte_carlo_agent import MonteCarloAgent
from agent.implementation.dqn.dqn_agent import DqnAgent


# エージェントファクトリークラス
class AgentFactory:
    # 与えられたエージェント名と環境から、エージェントのインスタンスを作成する
    @staticmethod
    def create_instance(name, env):
        if name == Agent.QLearning.value:
            return QAgent(env)
        elif name == Agent.MonteCarlo.value:
            return MonteCarloAgent(env)
        elif name == Agent.Sarsa.value:
            return SarsaAgent(env)
        elif name == Agent.Dqn.value:
            return DqnAgent(env)
        else:
            return None
