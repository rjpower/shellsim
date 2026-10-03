"""Heap queue algorithm over plain lists, with the inner loops of ``heapify`` and ``heappop``
in the ``_heapq`` native core."""

from _heapq import heapify, heappop

__all__ = [
    "heappush", "heappop", "heapify", "heapreplace", "merge", "nlargest", "nsmallest",
    "heappushpop", "heapify_max", "heappop_max", "heappush_max", "heappushpop_max",
    "heapreplace_max",
]


def _siftdown(heap, startpos, pos):
    newitem = heap[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


def _siftup(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    childpos = 2 * pos + 1
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    heap[pos] = newitem
    _siftdown(heap, startpos, pos)


def _siftdown_max(heap, startpos, pos):
    newitem = heap[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if parent < newitem:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


def _siftup_max(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    childpos = 2 * pos + 1
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos and not heap[rightpos] < heap[childpos]:
            childpos = rightpos
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    heap[pos] = newitem
    _siftdown_max(heap, startpos, pos)


def heappush(heap, item):
    heap.append(item)
    _siftdown(heap, 0, len(heap) - 1)


def heapreplace(heap, item):
    returnitem = heap[0]
    heap[0] = item
    _siftup(heap, 0)
    return returnitem


def heappushpop(heap, item):
    if heap and heap[0] < item:
        item, heap[0] = heap[0], item
        _siftup(heap, 0)
    return item


def heapify_max(heap):
    for index in reversed(range(len(heap) // 2)):
        _siftup_max(heap, index)


def heappop_max(heap):
    lastelt = heap.pop()
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup_max(heap, 0)
        return returnitem
    return lastelt


def heappush_max(heap, item):
    heap.append(item)
    _siftdown_max(heap, 0, len(heap) - 1)


def heapreplace_max(heap, item):
    returnitem = heap[0]
    heap[0] = item
    _siftup_max(heap, 0)
    return returnitem


def heappushpop_max(heap, item):
    if heap and item < heap[0]:
        item, heap[0] = heap[0], item
        _siftup_max(heap, 0)
    return item


def merge(*iterables, key=None, reverse=False):
    heap = []
    if reverse:
        push, pop, replace = heappush_max, heappop_max, heapreplace_max
    else:
        push, pop, replace = heappush, heappop, heapreplace
    for order, iterable in enumerate(iterables):
        iterator = iter(iterable)
        for value in iterator:
            sort_key = value if key is None else key(value)
            push(heap, (sort_key, order, value, iterator))
            break
    while len(heap) > 1:
        sort_key, order, value, iterator = heap[0]
        yield value
        advanced = False
        for value in iterator:
            sort_key = value if key is None else key(value)
            replace(heap, (sort_key, order, value, iterator))
            advanced = True
            break
        if not advanced:
            pop(heap)
    if heap:
        sort_key, order, value, iterator = heap[0]
        yield value
        for value in iterator:
            yield value


def nsmallest(n, iterable, key=None):
    if n <= 0:
        return []
    return sorted(iterable, key=key)[:n]


def nlargest(n, iterable, key=None):
    if n <= 0:
        return []
    return sorted(iterable, key=key, reverse=True)[:n]
