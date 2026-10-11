"""Exercise local public behavior in the actual guest."""

from opentelemetry import baggage, context, trace

ctx = baggage.set_baggage("key", "value")
assert baggage.get_baggage("key", context=ctx) == "value"
assert baggage.get_baggage("key") is None
with trace.get_tracer("guest").start_as_current_span("operation") as span:
    assert not span.is_recording()
assert isinstance(context.get_current(), context.Context)
print("opentelemetry-api: guest probe passed")
