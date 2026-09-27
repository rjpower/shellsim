"""Portable ``warnings`` semantics: filters, actions, recording and caller attribution."""

# These tests exercise warn()'s default stacklevel on purpose.
# ruff: noqa: B028

import warnings


def warn_from_caller(message):
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def test_catch_warnings_records_messages_categories_and_lines():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("direct")
        warn_from_caller("attributed")
    assert [str(item.message) for item in caught] == ["direct", "attributed"]
    assert [item.category for item in caught] == [UserWarning, RuntimeWarning]
    assert isinstance(caught[0].message, UserWarning)
    # stacklevel=2 attributes the warning to the calling line, the one after the first warning.
    assert caught[1].lineno == caught[0].lineno + 1
    assert caught[0].filename == caught[1].filename


def test_default_action_shows_each_location_once():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for _ in range(3):
            warnings.warn("repeated", UserWarning)
        warnings.warn("repeated", UserWarning)
    assert len(caught) == 2


def test_once_action_ignores_the_location():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("once")
        warnings.warn("one")
        warnings.warn("one")
        warnings.warn("two")
    assert [str(item.message) for item in caught] == ["one", "two"]


def test_error_action_raises_the_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            warnings.warn("boom", RuntimeWarning)
        except RuntimeWarning as error:
            assert str(error) == "boom"
        else:
            raise AssertionError("warning was not raised")


def test_filters_match_message_category_and_order():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message="divide by zero")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.warn("Divide by zero encountered", RuntimeWarning)
        warnings.warn("old", DeprecationWarning)
        warnings.warn("kept", UserWarning)
    assert [str(item.message) for item in caught] == ["kept"]


def test_catch_warnings_restores_the_filters():
    before = list(warnings.filters)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert warnings.filters[0][0] == "ignore"
    assert warnings.filters == before


def test_warning_instances_and_invalid_arguments():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn(FutureWarning("instance"))
    assert caught[0].category is FutureWarning
    for call, error_type in [
        (lambda: warnings.warn("x", int), TypeError),
        (lambda: warnings.simplefilter("sometimes"), ValueError),
        (lambda: warnings.filterwarnings("ignore", category=str), TypeError),
    ]:
        try:
            call()
        except error_type:
            pass
        else:
            raise AssertionError("invalid warning arguments were accepted")


def test_formatwarning_uses_the_cpython_layout():
    text = warnings.formatwarning("careful", UserWarning, "script.py", 3, line="  x = 1\n")
    assert text == "script.py:3: UserWarning: careful\n  x = 1\n"
