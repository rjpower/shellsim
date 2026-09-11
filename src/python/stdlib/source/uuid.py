"""Deterministic UUID subset for simulated programs."""


class UUID:
    def __init__(self, value):
        self.hex = value.replace("-", "")

    def __str__(self):
        value = self.hex
        return value[:8] + "-" + value[8:12] + "-" + value[12:16] + "-" + value[16:20] + "-" + value[20:]

    def __repr__(self):
        return "UUID('" + str(self) + "')"


class _State:
    def __init__(self):
        self.value = 0


_state = _State()


def uuid4():
    _state.value += 1
    suffix = f"{_state.value:012d}"
    return UUID("00000000000040008000" + suffix)
