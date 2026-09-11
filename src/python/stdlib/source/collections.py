"""Common container types implemented over ordinary Python protocols."""

from _collections import defaultdict


class Counter:
    def __init__(self, iterable=None):
        self._counts = {}
        if iterable is not None:
            self.update(iterable)

    def __getitem__(self, key):
        return self._counts.get(key, 0)

    def __setitem__(self, key, value):
        self._counts[key] = value

    def __delitem__(self, key):
        del self._counts[key]

    def __contains__(self, key):
        return key in self._counts

    def __iter__(self):
        return self._counts.keys()

    def __len__(self):
        return len(self._counts)

    def __repr__(self):
        return "Counter(" + repr(self._counts) + ")"

    def update(self, iterable=None):
        if iterable is None:
            return
        if isinstance(iterable, dict):
            for key, value in iterable.items():
                self._counts[key] = self[key] + value
        else:
            for key in iterable:
                self._counts[key] = self[key] + 1

    def subtract(self, iterable=None):
        if iterable is None:
            return
        if isinstance(iterable, dict):
            for key, value in iterable.items():
                self._counts[key] = self[key] - value
        else:
            for key in iterable:
                self._counts[key] = self[key] - 1

    def elements(self):
        result = []
        for key, count in self._counts.items():
            if count > 0:
                result.extend([key] * count)
        return result

    def most_common(self, n=None):
        values = sorted(self._counts.items(), key=lambda item: item[1], reverse=True)
        if n is None:
            return values
        return values[:n]

    def total(self):
        return sum(self._counts.values())

    def keys(self):
        return self._counts.keys()

    def values(self):
        return self._counts.values()

    def items(self):
        return self._counts.items()

    def clear(self):
        self._counts = {}

    def copy(self):
        return Counter(self._counts)

    def __add__(self, other):
        result = Counter()
        for key in self:
            value = self[key] + other[key]
            if value > 0:
                result[key] = value
        for key in other:
            if key not in self and other[key] > 0:
                result[key] = other[key]
        return result

    def __sub__(self, other):
        result = Counter()
        for key in self:
            value = self[key] - other[key]
            if value > 0:
                result[key] = value
        return result


class deque:
    def __init__(self, iterable=None, maxlen=None):
        self._items = []
        self.maxlen = maxlen
        if iterable is not None:
            self.extend(iterable)

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return self._items

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        self._items[index] = value

    def __repr__(self):
        return "deque(" + repr(self._items) + ")"

    def _trim_right(self):
        if self.maxlen is not None:
            while len(self._items) > self.maxlen:
                self._items.pop()

    def _trim_left(self):
        if self.maxlen is not None:
            while len(self._items) > self.maxlen:
                self._items.pop(0)

    def append(self, value):
        self._items.append(value)
        self._trim_left()

    def appendleft(self, value):
        self._items = [value] + self._items
        self._trim_right()

    def extend(self, iterable):
        for value in iterable:
            self.append(value)

    def extendleft(self, iterable):
        for value in iterable:
            self.appendleft(value)

    def pop(self):
        return self._items.pop()

    def popleft(self):
        return self._items.pop(0)

    def clear(self):
        self._items = []

    def copy(self):
        return deque(self._items, self.maxlen)

    def count(self, value):
        return self._items.count(value)

    def remove(self, value):
        self._items.remove(value)

    def reverse(self):
        self._items.reverse()

    def rotate(self, amount=1):
        if len(self._items) == 0:
            return
        if amount > 0:
            for _ in range(amount % len(self._items)):
                self.appendleft(self._items.pop())
        else:
            for _ in range((-amount) % len(self._items)):
                self.append(self._items.pop(0))
