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
"""

import logging
from logging import DEBUG as LOG_DEBUG
from logging import ERROR as LOG_ERROR
from classes import DiscordBot
from plot_commands import PlotCommands
from constants import F1DATABOT_ERR_CHANNEL, F1DATABOT_TOKEN

# set up logging
logging.basicConfig(filename="errors.log", 
                    level=LOG_DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)

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

    async def setup_hook(self):
        """Set up the bot's command cogs and prepare for operation.

        Loads the PlotCommands cog which contains all the visualization commands.
        This method is called automatically when the bot starts up.

        Returns:
            None: Loads the required cogs into the bot.
        """
        print("Loading cogs...")
        await self.add_cog(PlotCommands(self))
        print('Cogs installed.')


bot = MyBot()
bot.logger = logger 

@bot.event
async def on_ready():
    """Handle the bot's ready event.

    Called when the bot successfully connects to Discord. Syncs the command tree
    to ensure all slash commands are available to users.

    Returns:
        None: Prints connection status and syncs commands.
    """
    await bot.wait_until_ready()
    print('We have logged in as {0.user}'.format(bot))

    try:
        synced = await bot.tree.sync()
        print(f"synced {len(synced)} commands")
    except Exception as e:
        print(e)


@bot.event
async def on_command_error(ctx, error):
    """Handle errors that occur during command execution.

    Logs errors to both the error log file and a designated Discord channel
    for monitoring and debugging purposes.

    Args:
        ctx: The command context where the error occurred.
        error: The error that was raised.

    Returns:
        None: Logs the error and sends it to the error channel.
    """
    error_message = f'Error occurred while processing command {ctx.command}: {error}'
    bot.logger.log(LOG_ERROR, msg=error_message)
    chann = bot.get_channel(F1DATABOT_ERR_CHANNEL)
    await chann.send(error_message)


bot.run(F1DATABOT_TOKEN)