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
        self._gauss_next = None

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

    def gauss(self, mu=0.0, sigma=1.0):
        import math

        value = self._gauss_next
        self._gauss_next = None
        if value is None:
            angle = math.tau * self.random()
            radius = math.sqrt(-2.0 * math.log(1.0 - self.random()))
            value = math.cos(angle) * radius
            self._gauss_next = math.sin(angle) * radius
        return mu + value * sigma

    def expovariate(self, lambd=1.0):
        import math

        return -math.log(1.0 - self.random()) / lambd

    def gammavariate(self, alpha, beta):
        import math

        if not alpha > 0.0 or not beta > 0.0:
            raise ValueError("gammavariate requires positive shape and scale")

        # Boost small shapes to at least one, then scale by an independent
        # uniform draw. Both draws use this instance's modeled generator.
        if alpha < 1.0:
            return self.gammavariate(alpha + 1.0, beta) * self.random() ** (1.0 / alpha)

        # Rejection sampling for the unit-scale gamma distribution. Draw its
        # own normal value so gauss()'s cached second sample stays untouched.
        # The VM charges every loop and call, even when no draw is accepted.
        shape = alpha - 1.0 / 3.0
        spread = 1.0 / math.sqrt(9.0 * shape)
        while True:
            angle = math.tau * self.random()
            radius = math.sqrt(-2.0 * math.log(1.0 - self.random()))
            normal = math.cos(angle) * radius
            base = 1.0 + spread * normal
            if base <= 0.0:
                continue
            cube = base * base * base
            uniform = self.random()
            if uniform == 0.0 or uniform < 1.0 - 0.0331 * normal ** 4:
                return beta * shape * cube
            if math.log(uniform) < 0.5 * normal * normal + shape * (1.0 - cube + math.log(cube)):
                return beta * shape * cube

    def betavariate(self, alpha, beta):
        if not alpha > 0.0 or not beta > 0.0:
            raise ValueError("betavariate requires positive shapes")
        left = self.gammavariate(alpha, 1.0)
        right = self.gammavariate(beta, 1.0)
        if left == 0.0 and right == 0.0:
            return 0.0
        if left == right == float('inf'):
            return 0.5
        if left > right:
            return 1.0 / (1.0 + right / left)
        return left / right / (1.0 + left / right)

    def weibullvariate(self, alpha, beta):
        import math

        return alpha * (-math.log(1.0 - self.random())) ** (1.0 / beta)

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


def gauss(mu=0.0, sigma=1.0):
    return _default.gauss(mu, sigma)


def expovariate(lambd=1.0):
    return _default.expovariate(lambd)


def gammavariate(alpha, beta):
    return _default.gammavariate(alpha, beta)


def betavariate(alpha, beta):
    return _default.betavariate(alpha, beta)


def weibullvariate(alpha, beta):
    return _default.weibullvariate(alpha, beta)


def choice(population):
    return _default.choice(population)


def shuffle(values):
    return _default.shuffle(values)


def sample(population, count):
    return _default.sample(population, count)
