//! Observable compatibility tests for shellsim's cooperative asyncio subset.

use super::support::run_python_text;

#[test]
fn async_function_is_lazy_and_returns_through_await() {
    let source = r#"import asyncio
events = []
async def inner(value):
    events.append("inner")
    return value * 2
async def outer():
    events.append("outer")
    return await inner(21)
coroutine = outer()
print(events)
print(asyncio.run(coroutine))
print(events)
"#;
    let (status, stdout, stderr) = run_python_text(source);
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "[]\n42\n['outer', 'inner']\n");
}

#[test]
fn gather_interleaves_at_sleep_zero_and_preserves_result_order() {
    let source = r#"import asyncio
events = []
async def worker(name):
    events.append(name + "1")
    await asyncio.sleep(0)
    events.append(name + "2")
    return name
async def main():
    return await asyncio.gather(worker("a"), worker("b"))
print(asyncio.run(main()))
print(events)
"#;
    let (status, stdout, stderr) = run_python_text(source);
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "['a', 'b']\n['a1', 'b1', 'a2', 'b2']\n");
}

#[test]
fn blocked_event_waiter_does_not_prevent_its_producer_from_running() {
    let source = r#"import asyncio
events = []
async def main():
    ready = asyncio.Event()
    async def consumer():
        events.append("waiting")
        await ready.wait()
        events.append("awake")
    async def producer():
        events.append("produce")
        ready.set()
    consumer_task = asyncio.create_task(consumer())
    producer_task = asyncio.create_task(producer())
    await producer_task
    await consumer_task
asyncio.run(main())
print(events)
"#;
    let (status, stdout, stderr) = run_python_text(source);
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "['waiting', 'produce', 'awake']\n");
}

#[test]
fn logical_timers_and_queue_wakeups_preserve_async_ordering() {
    let source = r#"import asyncio
events = []
async def delayed(name, delay):
    await asyncio.sleep(delay)
    events.append(name)
async def main():
    queue = asyncio.Queue()
    async def consumer():
        events.append(await queue.get())
    consumer_task = asyncio.create_task(consumer())
    await asyncio.sleep(0)
    await queue.put("queued")
    await asyncio.gather(delayed("later", 2), delayed("sooner", 1))
    await consumer_task
asyncio.run(main())
print(events)
"#;
    let (status, stdout, stderr) = run_python_text(source);
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "['queued', 'sooner', 'later']\n");
}

#[test]
fn async_with_lock_serializes_critical_sections() {
    let source = r#"import asyncio
events = []
async def main():
    lock = asyncio.Lock()
    async def worker(name):
        async with lock:
            events.append(name + " enter")
            await asyncio.sleep(0)
            events.append(name + " exit")
    await asyncio.gather(worker("a"), worker("b"))
asyncio.run(main())
print(events)
"#;
    let (status, stdout, stderr) = run_python_text(source);
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "['a enter', 'a exit', 'b enter', 'b exit']\n");
}
