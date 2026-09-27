# Portable checks of contextlib's context managers and ExitStack callback ordering.

from contextlib import ExitStack, closing, contextmanager, nullcontext, suppress


def test_contextmanager_runs_setup_and_cleanup_around_the_body():
    events = []

    @contextmanager
    def tag(name):
        events.append(f"<{name}>")
        try:
            yield name.upper()
        finally:
            events.append(f"</{name}>")

    with tag("a") as value:
        events.append(value)
    assert events == ["<a>", "A", "</a>"]

    try:
        with tag("b"):
            raise ValueError("boom")
    except ValueError as error:
        assert str(error) == "boom"
    assert events[-1] == "</b>"


def test_contextmanager_can_suppress_or_replace_an_exception():
    @contextmanager
    def swallow():
        try:
            yield
        except KeyError:
            pass

    with swallow():
        raise KeyError("ignored")

    @contextmanager
    def translate():
        try:
            yield
        except KeyError as error:
            raise LookupError("translated") from error

    try:
        with translate():
            raise KeyError("original")
    except LookupError as error:
        assert str(error) == "translated"
    else:
        raise AssertionError("the replacement exception was not raised")


def test_contextmanager_rejects_generators_that_do_not_yield_once():
    @contextmanager
    def silent():
        if False:
            yield

    @contextmanager
    def twice():
        yield
        yield

    for factory, message in ((silent, "generator didn't yield"), (twice, "generator didn't stop")):
        try:
            with factory():
                pass
        except RuntimeError as error:
            assert str(error) == message
        else:
            raise AssertionError(message)


def test_suppress_nullcontext_and_closing():
    with suppress(KeyError, IndexError):
        [][1]
    try:
        with suppress(KeyError):
            raise ValueError("kept")
    except ValueError as error:
        assert str(error) == "kept"
    with nullcontext(5) as value:
        assert value == 5

    closed = []

    class Resource:
        def close(self):
            closed.append(True)

    with closing(Resource()) as resource:
        assert isinstance(resource, Resource)
    assert closed == [True]


def test_exit_stack_unwinds_in_reverse_and_passes_exceptions_along():
    order = []

    @contextmanager
    def entered(name):
        order.append(f"enter {name}")
        yield name
        order.append(f"exit {name}")

    with ExitStack() as stack:
        stack.callback(order.append, "first callback")
        assert stack.enter_context(entered("cm")) == "cm"
        stack.callback(order.append, "last callback")
    assert order == ["enter cm", "last callback", "exit cm", "first callback"]

    seen = []

    def suppressor(kind, value, traceback):
        seen.append(kind)
        return True

    with ExitStack() as stack:
        stack.push(suppressor)
        raise KeyError("suppressed")
    assert seen == [KeyError]

    try:
        with ExitStack() as stack:
            stack.callback(order.append, "still runs")

            def fail():
                raise IndexError("from callback")

            stack.callback(fail)
    except IndexError as error:
        assert str(error) == "from callback"
    assert order[-1] == "still runs"

    stack = ExitStack()
    stack.callback(order.append, "moved")
    moved = stack.pop_all()
    stack.close()
    assert order[-1] == "still runs"
    moved.close()
    assert order[-1] == "moved"

    try:
        ExitStack().enter_context(object())
    except TypeError:
        pass
    else:
        raise AssertionError("enter_context accepted an object without __enter__")
