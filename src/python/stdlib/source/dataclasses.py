"""Data classes over shellsim's class annotations and metered VM.

The class marker lives in ``_dataclasses`` so construction can fill annotated fields
without evaluating source or granting access to host Python.
"""

from _dataclasses import mark_dataclass as _mark_dataclass


class _Missing:
    pass


MISSING = _Missing()


class FrozenInstanceError(AttributeError):
    pass


class Field:
    _shellsim_dataclass_field = True

    def __init__(self, default=MISSING, default_factory=MISSING):
        self.default = default
        self.default_factory = default_factory
        self.has_default = default is not MISSING
        self.has_factory = default_factory is not MISSING


def field(*, default=MISSING, default_factory=MISSING):
    """Declare a default value or a factory called for each new instance."""
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("cannot specify both default and default_factory")
    return Field(default, default_factory)


def dataclass(cls=None, *, eq=True, frozen=False, kw_only=False):
    """Generate the common field initializer and value protocols for a class."""
    def decorate(target):
        _mark_dataclass(target)
        target.__shellsim_dataclass__ = True
        target.__shellsim_dataclass_kw_only__ = kw_only
        names = target.__shellsim_dataclass_field_names__

        if eq:
            def __eq__(self, other):
                if type(self) is not type(other):
                    return NotImplemented
                return tuple(getattr(self, name) for name in names) == tuple(
                    getattr(other, name) for name in names
                )

            target.__eq__ = __eq__
            if not frozen:
                target.__hash__ = None

        if frozen:
            def reject_assignment(self, name, value):
                raise FrozenInstanceError("cannot assign to field " + repr(name))

            def reject_deletion(self, name):
                raise FrozenInstanceError("cannot delete field " + repr(name))

            target.__setattr__ = reject_assignment
            target.__delattr__ = reject_deletion
            if eq:
                def __hash__(self):
                    return hash(tuple(getattr(self, name) for name in names))

                target.__hash__ = __hash__

        return target

    return decorate if cls is None else decorate(cls)


def is_dataclass(obj):
    """Whether a class or instance has been decorated as a data class."""
    cls = obj if isinstance(obj, type) else type(obj)
    return bool(getattr(cls, "__shellsim_dataclass__", False))


def fields(obj):
    raise NotImplementedError("dataclasses.fields is not implemented")
