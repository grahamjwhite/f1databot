import discord
import logging
from discord.ext import commands


class DiscordBot(commands.Bot):
    """A Discord bot for Formula 1 data analysis and visualization.

    This class extends discord.ext.commands.Bot to create a specialized bot for
    Formula 1 data analysis. It handles command processing and interaction with
    Discord's API.

    Attributes:
        logger (logging.Logger): Logger instance for tracking bot operations and errors.
    """

    logger: logging.Logger

    def __init__(self) -> None:
        """Initialize the Discord bot with custom prefix and intents.

        Sets up the bot with:
        - Custom command prefix '$f1databot'
        - Default Discord intents
        - Message content intent (currently disabled)
        """
        command_prefix = '$f1databot'
        intents = discord.Intents.default()
        #intents.message_content = True

        super().__init__(command_prefix, intents=intents)


class ValidationException(Exception):
    """Custom exception for handling validation errors in the application.

    This exception is raised when input validation fails or when data
    does not meet expected criteria. It extends the base Exception class
    to provide specific error handling for validation-related issues.

    Args:
        *args: Variable length argument list passed to the parent Exception class.
    """

    def __init__(self, *args: object) -> None:
        """Initialize the validation exception.

        Args:
            *args: Variable length argument list containing the error message
                and any additional information about the validation failure.
        """
        super().__init__(*args)
