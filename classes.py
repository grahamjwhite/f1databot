import discord
import logging
from discord.ext import commands


class DiscordBot(commands.Bot):

    logger: logging.Logger

    def __init__(self) -> None:
        command_prefix = '$f1databot'
        intents = discord.Intents.default()
        #intents.message_content = True

        super().__init__(command_prefix, intents=intents)


class ValidationException(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
