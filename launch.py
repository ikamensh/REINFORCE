import random
import math

from short_coridor import ShortCoridorEnv


env = ShortCoridorEnv()

def evaluate(p_right: float) -> float:
    """evaluate policy many times in the ShortCoridorEnv"""
    trials = 100_000
    total_rew = 0
    for i in range(trials):
        env.reset()
        done = False
        while not done:
            action = 1 if random.random() < p_right else 0
            o, r, done, _ = env.step(action)
            total_rew += r

    return total_rew / trials

best = evaluate(0.59)
print(best)

left_eps = evaluate(0.05)
print(left_eps)

right_eps = evaluate(0.95)
print(right_eps)
