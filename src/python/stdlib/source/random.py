"""Deterministic pseudo-random helpers without host entropy or clock access."""


class Random:
    def __init__(self, value=None):
        self.seed(value)

    def seed(self, value=None):
        if value is None:
            value = 0
        if not isinstance(value, int):
            raise TypeError("shellsim random seeds must be integers or None")
        self._state = value % 2147483648

    def _next(self):
        self._state = (self._state * 1103515245 + 12345) % 2147483648
        return self._state

    def random(self):
        return self._next() / 2147483648

    def randrange(self, start, stop=None, step=1):
        if stop is None:
            stop = start
            start = 0
        if step == 0:
            raise ValueError("zero step for randrange()")
        width = stop - start
        if step > 0:
            count = (width + step - 1) // step
        else:
            count = (width + step + 1) // step
        if count <= 0:
            raise ValueError("empty range for randrange()")
        return start + step * (self._next() % count)

    def randint(self, left, right):
        return self.randrange(left, right + 1)

    def uniform(self, left, right):
        return left + (right - left) * self.random()

    def choice(self, population):
        if len(population) == 0:
            raise IndexError("cannot choose from an empty sequence")
        return population[self.randrange(len(population))]

    def shuffle(self, values):
        for index in range(len(values) - 1, 0, -1):
            selected = self.randrange(index + 1)
            value = values[index]
            values[index] = values[selected]
            values[selected] = value

    def sample(self, population, count):
        if count < 0 or count > len(population):
            raise ValueError("sample larger than population or is negative")
        available = list(population)
        result = []
        for _ in range(count):
            result.append(available.pop(self.randrange(len(available))))
        return result


_default = Random(0)


def seed(value=None):
    return _default.seed(value)


def random():
    return _default.random()


def randrange(start, stop=None, step=1):
    return _default.randrange(start, stop, step)


def randint(left, right):
    return _default.randint(left, right)


def uniform(left, right):
    return _default.uniform(left, right)


def choice(population):
    return _default.choice(population)


def shuffle(values):
    return _default.shuffle(values)


def sample(population, count):
    return _default.sample(population, count)
