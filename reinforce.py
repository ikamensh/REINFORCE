import random

import torch as t
from pennpaper import Metric, plot_group

from short_coridor import ShortCoridorEnv


class CoridorAgent:
    def __init__(self):
        self.param = t.tensor(6 * (random.random() - 0.5), requires_grad=True)

    def policy(self) -> int:
        if random.random() < self.prob(1):
            return 1
        else:
            return 0

    def prob(self, action) -> t.Tensor:
        assert action in (0, 1)

        prob1 = t.sigmoid(self.param)
        if action == 0:
            return 1 - prob1
        else:
            return prob1


def reinforce(lr: float):
    assert 0 < lr < 1

    m = Metric(f"lr={lr:.5f}")
    env = ShortCoridorEnv()
    agent = CoridorAgent()

    for episode in range(1_000):
        history, total_reward = rollout(agent, env)
        m.add_record(episode, total_reward)
        G = sum(r for o, a, r in history)
        for o, a, r in history:
            apply_grad(G, a, agent, lr)
            G -= r
    return m


def rollout(agent, env):
    env.reset()
    done = False
    history = []
    total_reward = 0
    while not done:
        with t.no_grad():
            action = agent.policy()
        obs, rew, done, _ = env.step(action)
        total_reward += rew
        history.append((obs, action, rew))
    return history, total_reward


def apply_grad(G, a, agent, lr):
    if agent.param.grad is not None:
        agent.param.grad.zero_()
    prob = agent.prob(a)
    prob.backward()
    with t.no_grad():
        update = lr * G * agent.param.grad / (prob + 1e-10)
        agent.param += update


import time

start = time.time()

random.seed(0)
m1 = sum(reinforce(2 ** -13) for _ in range(10))
random.seed(0)
m2 = sum(reinforce(2 ** -12) for _ in range(10))
random.seed(0)
m3 = sum(reinforce(2 ** -14) for _ in range(10))

plot_group(
    [
        m1,
        m2,
        m3,
    ],
    smoothen=True,
    stdev_factor=0.3,
    name=f"reinforce_coridor_{time.time() - start:.3f}",
)
