"""Configuration constants for the Formula 1 data analysis Discord bot.

This module defines global constants used throughout the application.
These constants are primarily sourced from environment variables and provide
configuration values for:

- Caching paths (plots and data)
- Discord bot token
- Discord error channel ID

Usage:
    Import required constants from this module where needed.

    Example:
    from constants import F1DATABOT_TOKEN
    bot.run(F1DATABOT_TOKEN)

Environment Variables:
    - F1DATABOT_PLOT_CACHE: Path to the directory for storing plot caches.
    - F1DATABOT_DATA_CACHE: Path to the directory for storing FastF1 data caches.
    - F1DATABOT_TOKEN: Discord bot token for authentication.
    - F1DATABOT_ERR_CHANNEL: Discord channel ID for sending error messages.
"""

import os

F1DATABOT_PLOT_CACHE: str = os.environ['F1DATABOT_PLOT_CACHE']
F1DATABOT_DATA_CACHE: str = os.environ["F1DATABOT_DATA_CACHE"]
F1DATABOT_TOKEN: str = os.environ["F1DATABOT_TOKEN"]
F1DATABOT_ERR_CHANNEL: int = int(os.environ["F1DATABOT_ERR_CHANNEL"])

