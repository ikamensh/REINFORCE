"""Implements short coridor environment per RL by Sutton, 2nd Ed. p.323"""


class States:
    start = 0
    one = 1
    two = 2
    end = 3

s = States


class ShortCoridorEnv:

    def __init__(self):
        self._state = None

    def reset(self):
        self._state = s.start
        return None

    def step(self, action: int):
        """0 is left, 1 is right"""
        assert action in (0, 1)
        assert self._state in (s.start, s.one, s.two)

        action = action * 2 - 1

        if self._state == s.one:
            action *= -1

        self._state += action
        self._state = max(s.start, self._state)
        done = self._state == s.end
        return None, -1, done, {}
