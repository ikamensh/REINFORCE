import math
import random

from pennpaper import Metric, plot, plot_group

from short_coridor import ShortCoridorEnv

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


class CoridorAgent:

    def __init__(self):
        self.param = 6 * ( random.random() - 0.5 )

    def policy(self) -> int:
        prob_1 = sigmoid(self.param)
        if random.random() < prob_1:
            return 1
        else:
            return 0

    def grad(self, action) -> float:
        assert action in (0, 1)

        s = sigmoid(self.param)
        sigm_grad = s * (1-s)
        if action == 0:
            return -sigm_grad
        else:
            return sigm_grad



def reinforce(lr: float):
    assert 0 < lr < 1

    m = Metric(f"lr={lr:.5f}")
    env = ShortCoridorEnv()
    agent = CoridorAgent()

    for episode in range(3_000):
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
        action = agent.policy()
        obs, rew, done, _ = env.step(action)
        total_reward += rew
        history.append((obs, action, rew))
    return history, total_reward


def apply_grad(G, a, agent, lr):
    prob_1 = sigmoid(agent.param)
    prob = prob_1 if a == 1 else 1 - prob_1
    update = lr * G * agent.grad(a) / ( prob + 1e-10)
    agent.param += update


m1 = sum( reinforce(2 ** -13) for _ in range(100) )
m2 = sum( reinforce(2 ** -12) for _ in range(100) )
m3 = sum( reinforce(2 ** -14) for _ in range(100) )
m4 = sum( reinforce(2 ** -15) for _ in range(100) )
m5 = sum( reinforce(2 ** -11) for _ in range(100) )
# m6 = sum( reinforce(2 ** -10) for _ in range(100) )

plot_group([m1, m2, m3, m4, m5,], smoothen=True, stdev_factor=0.3, name="reinforce_coridor")
