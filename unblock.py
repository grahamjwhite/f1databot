import functools
import typing
import asyncio


def to_thread(func: typing.Callable) -> typing.Coroutine:
    """Decorator to run a synchronous function in a separate thread.

    This decorator allows synchronous functions to be run asynchronously by
    executing them in a separate thread, preventing blocking of the main event loop.
    Useful for CPU-intensive operations or I/O-bound tasks.

    Args:
        func (typing.Callable): The synchronous function to be run in a thread.

    Returns:
        typing.Coroutine: An awaitable coroutine that will run the function in a thread.

    Example:
        @to_thread
        def cpu_intensive_function():
            # This will run in a separate thread
            pass
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper

