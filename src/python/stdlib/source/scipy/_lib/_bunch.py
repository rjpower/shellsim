"""Named result classes, following ``scipy/_lib/_bunch.py`` (SciPy 1.18).

SciPy's results are ``tuple`` subclasses built by ``namedtuple`` or ``_make_tuple_bunch``: the
named fields unpack and index like a tuple, and a "bunch" may carry extra fields that are only
attributes. shellsim cannot subclass ``tuple``, so these classes unpack, index, compare and
print the same way but are not ``tuple`` instances, and their fields can be reassigned.
"""

__all__ = ["_make_tuple_bunch"]


class _TupleBunch:
    """Base class for the results ``_make_tuple_bunch`` creates."""

    _fields = ()
    _extra_fields = ()

    def __init__(self, *args, **kwargs):
        name = type(self).__name__
        fields = self._fields
        if len(args) > len(fields):
            raise TypeError(
                f"{name}() takes {len(fields)} positional arguments but {len(args)} were given"
            )
        values = list(args)
        for field in fields[len(args) :]:
            if field not in kwargs:
                raise TypeError(f"{name}() missing required argument: '{field}'")
            values.append(kwargs.pop(field))
        for key in self._extra_fields:
            if key not in kwargs:
                raise TypeError(f"missing keyword argument '{key}'")
        for key in kwargs:
            if key not in self._extra_fields:
                raise TypeError(f"unexpected keyword argument '{key}'")
        self._values = tuple(values)
        for field, value in zip(fields, values):
            setattr(self, field, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, index):
        return self._values[index]

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def _asdict(self):
        return {name: getattr(self, name) for name in self._fields + self._extra_fields}

    def __repr__(self):
        fields = ", ".join(f"{name}={value!r}" for name, value in self._asdict().items())
        return f"{type(self).__name__}({fields})"


def _make_tuple_bunch(typename, field_names, extra_field_names=None):
    """Create a result class with tuple fields ``field_names`` and attribute-only extras.

    ``namedtuple(typename, field_names)`` is the special case with no extra fields.
    """
    if len(field_names) == 0:
        raise ValueError("field_names must contain at least one name")
    extra_field_names = () if extra_field_names is None else tuple(extra_field_names)
    return type(
        typename,
        (_TupleBunch,),
        {"_fields": tuple(field_names), "_extra_fields": extra_field_names},
    )
