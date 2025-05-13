"""Utilities for running synchronous code asynchronously.

This module provides tools for running CPU-intensive or I/O-bound synchronous
code in separate threads to prevent blocking the main event loop. It's particularly
useful for data processing and visualization tasks that might otherwise block
the Discord bot's responsiveness.

The module provides:
- to_thread decorator: Converts synchronous functions to run in separate threads
- Thread management for async operations
- Integration with asyncio event loop

Dependencies:
    - asyncio: For async/await support
    - functools: For function wrapping
    - typing: For type hints
"""

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

