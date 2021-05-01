import random
import math

from short_coridor import ShortCoridorEnv


env = ShortCoridorEnv()

def evaluate(p_right: float) -> float:
    """evaluate policy many times in the ShortCoridorEnv"""
    trials = 200_000
    total_rew = 0
    for i in range(trials):
        env.reset()
        done = False
        while not done:
            action = 1 if random.random() < p_right else 0
            o, r, done, _ = env.step(action)
            total_rew += r

    return total_rew / trials

def test_avg_rew_as_expected():
    best = evaluate(0.59)
    assert math.isclose(best, -11.6, rel_tol=0.01)

    left_eps = evaluate(0.05)
    assert math.isclose(left_eps, -82, rel_tol=0.03)

    right_eps = evaluate(0.95)
    assert math.isclose(right_eps, -44, rel_tol=0.03)
