//! Observable compatibility tests for shellsim's cooperative asyncio subset.

use super::support::run_python_text;
use shellsim::Environment;

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

#[test]
fn task_exceptions_propagate_and_gather_can_return_them() {
    let source = r#"import asyncio
async def fail():
    await asyncio.sleep(0)
    raise ValueError("boom")
async def main():
    task = asyncio.create_task(fail())
    try:
        await task
    except ValueError as error:
        print(str(error))
    results = await asyncio.gather(fail(), return_exceptions=True)
    print(isinstance(results[0], ValueError))
asyncio.run(main())
"#;
    let (status, stdout, stderr) = run_python_text(source);
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "boom\nTrue\n");
}

#[test]
fn cancellation_and_wait_for_run_finally_cleanup() {
    let source = r#"import asyncio
events = []
async def blocked(label):
    try:
        await asyncio.Event().wait()
    finally:
        events.append(label)
async def main():
    task = asyncio.create_task(blocked("cancel"))
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print(task.done(), task.cancelled())
    try:
        await asyncio.wait_for(blocked("timeout"), 1)
    except TimeoutError:
        print("timed out")
asyncio.run(main())
print(events)
"#;
    let (status, stdout, stderr) = run_python_text(source);
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "True True\ntimed out\n['cancel', 'timeout']\n");
}

#[test]
fn subprocess_streams_make_progress_across_modeled_resource_waits() {
    let source = r#"import asyncio
async def main():
    process = await asyncio.create_subprocess_exec(
        "cat", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE
    )
    output, error = await process.communicate(b"x" * 100000)
    print(process.returncode, len(output), error)
asyncio.run(main())
"#;
    let (status, stdout, stderr) = run_python_text(source);
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "0 100000 None\n");
}

#[test]
fn subprocess_wait_timeout_uses_virtual_time_and_remains_recoverable() {
    let source = r#"import asyncio
async def main():
    process = await asyncio.create_subprocess_exec("sleep", "10")
    try:
        await asyncio.wait_for(process.wait(), 1)
    except TimeoutError:
        print("timeout")
        process.kill()
    print(await process.wait())
asyncio.run(main())
"#;
    let (status, stdout, stderr) = run_python_text(source);
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "timeout\n-9\n");
}

#[test]
fn scheduled_python_wakes_when_any_async_child_resource_is_ready() {
    let source = r#"import asyncio
async def main():
    process = await asyncio.create_subprocess_exec(
        "cat", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE
    )
    output, error = await process.communicate(b"x" * 100000)
    print(process.returncode, len(output), error)
asyncio.run(main())
"#;
    let mut environment = Environment::new();
    environment
        .vfs
        .put_file("/async.py", source.as_bytes().to_vec(), 0o644)
        .expect("install async process script");
    let (outcome, stdout, stderr) = environment.run_script_capture("python3.14 /async.py");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(String::from_utf8(stdout).unwrap(), "0 100000 None\n");
    assert!(stderr.is_empty());
}
