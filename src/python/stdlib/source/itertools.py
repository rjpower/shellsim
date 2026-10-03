"""Iterator building blocks. The combinatoric generators and ``islice``/``count`` live in the
``_itertools`` native core; the rest are ordinary generators so every produced item is a metered
instruction."""

from _itertools import combinations, count, islice, permutations, product
import _itertools


def chain(*iterables):
    return _itertools.chain(*iterables)


def _chain_from_iterable(iterables):
    for iterable in iterables:
        yield from iterable


chain.from_iterable = _chain_from_iterable


def accumulate(iterable, func=None, *, initial=None):
    iterator = iter(iterable)
    total = initial
    if initial is None:
        for total in iterator:
            break
        else:
            return
    yield total
    for element in iterator:
        total = total + element if func is None else func(total, element)
        yield total


def batched(iterable, n, *, strict=False):
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while True:
        batch = tuple(islice(iterator, n))
        if not batch:
            return
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def combinations_with_replacement(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    if not n and r:
        return
    indices = [0] * r
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != n - 1:
                break
        else:
            return
        indices[i:] = [indices[i] + 1] * (r - i)
        yield tuple(pool[i] for i in indices)


def compress(data, selectors):
    return (datum for datum, selector in zip(data, selectors) if selector)


def cycle(iterable):
    saved = []
    for element in iterable:
        yield element
        saved.append(element)
    while saved:
        for element in saved:
            yield element


def dropwhile(predicate, iterable):
    iterator = iter(iterable)
    for x in iterator:
        if not predicate(x):
            yield x
            break
    for x in iterator:
        yield x


def filterfalse(predicate, iterable):
    if predicate is None:
        predicate = bool
    for x in iterable:
        if not predicate(x):
            yield x


class groupby:
    """Group consecutive elements of ``iterable`` sharing the same ``key(element)``."""

    def __init__(self, iterable, key=None):
        self._keyfunc = key if key is not None else (lambda value: value)
        self._iterator = iter(iterable)
        self._exhausted = False
        self._tgtkey = self._currkey = self._currvalue = object()
        self._marker = self._tgtkey
        self._id = 0

    def __iter__(self):
        return self

    def __next__(self):
        self._id += 1
        while self._currkey is self._marker or self._currkey == self._tgtkey:
            if self._exhausted:
                raise StopIteration
            try:
                self._currvalue = next(self._iterator)
            except StopIteration:
                self._exhausted = True
                raise
            self._currkey = self._keyfunc(self._currvalue)
            if self._tgtkey is self._marker:
                break
        self._tgtkey = self._currkey
        return (self._currkey, self._grouper(self._tgtkey, self._id))

    def _grouper(self, tgtkey, group_id):
        while self._id == group_id and self._currkey == tgtkey:
            yield self._currvalue
            if self._exhausted:
                return
            try:
                self._currvalue = next(self._iterator)
            except StopIteration:
                self._exhausted = True
                self._currkey = self._marker
                return
            self._currkey = self._keyfunc(self._currvalue)


def pairwise(iterable):
    iterator = iter(iterable)
    previous = None
    for previous in iterator:
        break
    else:
        return
    for current in iterator:
        yield (previous, current)
        previous = current


class repeat:
    def __init__(self, object, times=None):
        self._object = object
        self._times = times
        if times is not None and times < 0:
            self._times = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._times is None:
            return self._object
        if self._times <= 0:
            raise StopIteration
        self._times -= 1
        return self._object

    def __length_hint__(self):
        return 0 if self._times is None else self._times

    def __repr__(self):
        if self._times is None:
            return "repeat(" + repr(self._object) + ")"
        return "repeat(" + repr(self._object) + ", " + repr(self._times) + ")"


def starmap(function, iterable):
    for args in iterable:
        yield function(*args)


def takewhile(predicate, iterable):
    for x in iterable:
        if predicate(x):
            yield x
        else:
            break


class _TeeBuffer:
    def __init__(self, iterator):
        self.iterator = iterator
        self.values = []
        self.exhausted = False


class _Tee:
    def __init__(self, buffer, index=0):
        self._buffer = buffer
        self._index = index

    def __iter__(self):
        return self

    def __next__(self):
        buffer = self._buffer
        if self._index == len(buffer.values):
            if buffer.exhausted:
                raise StopIteration
            try:
                buffer.values.append(next(buffer.iterator))
            except StopIteration:
                buffer.exhausted = True
                raise
        value = buffer.values[self._index]
        self._index += 1
        return value

    def __copy__(self):
        return _Tee(self._buffer, self._index)


def tee(iterable, n=2):
    if n < 0:
        raise ValueError("n must be >= 0")
    buffer = _TeeBuffer(iter(iterable))
    return tuple(_Tee(buffer) for _ in range(n))


def zip_longest(*iterables, fillvalue=None):
    iterators = [iter(iterable) for iterable in iterables]
    active = len(iterators)
    if not active:
        return
    while True:
        values = []
        for index, iterator in enumerate(iterators):
            try:
                value = next(iterator)
            except StopIteration:
                active -= 1
                if not active:
                    return
                iterators[index] = repeat(fillvalue)
                value = fillvalue
            values.append(value)
        yield tuple(values)
