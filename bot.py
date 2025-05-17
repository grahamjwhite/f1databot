"""Main entry point for the Formula 1 data analysis Discord bot.

This module initializes and runs the Discord bot that provides Formula 1 data
analysis and visualization capabilities. It handles bot setup, command loading,
error handling, and logging.

The module:
- Sets up logging configuration
- Initializes the bot with required intents
- Loads command cogs
- Handles bot events (ready, errors)
- Manages command synchronization

Dependencies:
    - discord.py: For Discord bot functionality
    - logging: For error and operation logging
    - plot_commands: For visualization commands
    - classes: For bot class definitions
    - constants: For token and other configurations
    - asyncio: For running the bot asynchronously
"""

import logging
import asyncio
import discord
from logging import DEBUG as LOG_DEBUG
from logging import ERROR as LOG_ERROR
from classes import DiscordBot
from plot_commands import PlotCommands
from constants import F1DATABOT_ERR_CHANNEL, F1DATABOT_TOKEN

# set up logging
LOG_FORMAT = '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'
logging.basicConfig(level=LOG_DEBUG, 
                    format=LOG_FORMAT, 
                    datefmt='%d-%m-%y %H:%M:%S',
                    filename='f1databot.log')

# Mute discord loggers below WARNING level
logging.getLogger('discord.http').setLevel(logging.WARNING)
logging.getLogger('discord.client').setLevel(logging.WARNING)
logging.getLogger('discord.gateway').setLevel(logging.WARNING)

class MyBot(DiscordBot):
    """A specialized Discord bot for Formula 1 data analysis and visualization.

    This class extends the base DiscordBot to add Formula 1 specific functionality,
    including command handling, error logging, and cog management.

    Attributes:
        logger (logging.Logger): Logger instance for tracking bot operations and errors.
    """
        
    def __init__(self) -> None:
        """Initialize the Formula 1 data bot.

        Sets up the bot with default configuration and prepares it for command handling.
        """
        super().__init__()
        self.logger: logging.Logger = logging.getLogger(type(self).__name__)

    async def setup_hook(self) -> None:
        """Set up the bot's command cogs and prepare for operation.

        Loads the PlotCommands cog which contains all the visualization commands.
        This method is called automatically when the bot starts up.
        """
        self.logger.info("Setup hook running...")
        await self.add_cog(PlotCommands(self))
        self.logger.info("PlotCommands cog added.")
        # Optional: Sync commands selectively or globally
        # Synced = await self.tree.sync() # Sync globally
        # self.logger.info(f"Synced {len(Synced)} commands globally.")

    async def on_ready(self) -> None:
        """Called when the bot is fully connected and ready.

        Logs bot readiness and basic information.
        Also syncs the application command tree with Discord.
        """
        self.logger.info(f"Logged in as {self.user.name} (ID: {self.user.id})")
        self.logger.info(f"Discord.py version: {discord.__version__}")
        self.logger.info("Bot is ready and online!")
        
        try:
            synced = await self.tree.sync()
            self.logger.info(f"Synced {len(synced)} commands globally.")
        except Exception as e:
            self.logger.error(f"Error syncing command tree: {e}", exc_info=True)


def main() -> None:
    """Creates the bot instance and runs it.

    Initializes the MyBot class and starts the bot's event loop using the token
    from environment variables.
    """
    bot: MyBot = MyBot()
    try:
        bot.run(F1DATABOT_TOKEN, log_handler=None) # Use built-in handler or None if basicConfig is enough
    except discord.errors.LoginFailure:
        bot.logger.critical("Failed to log in: Invalid Discord token provided.")
    except Exception as e:
        bot.logger.critical(f"Critical error during bot execution: {e}", exc_info=True)

if __name__ == '__main__':
    main()