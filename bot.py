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
        
    def __init__(self) -> None:
        super().__init__()

    async def setup_hook(self):
        print("Loading cogs...")
        await self.add_cog(PlotCommands(self))
        print('Cogs installed.')


bot = MyBot()
bot.logger = logger 

@bot.event
async def on_ready():
    await bot.wait_until_ready()
    print('We have logged in as {0.user}'.format(bot))

    try:
        synced = await bot.tree.sync()
        print(f"synced {len(synced)} commands")
    except Exception as e:
        print(e)


@bot.event
async def on_command_error(ctx, error):
    error_message = f'Error occurred while processing command {ctx.command}: {error}'
    bot.logger.log(LOG_ERROR, msg=error_message)
    chann = bot.get_channel(F1DATABOT_ERR_CHANNEL)
    await chann.send(error_message)


bot.run(F1DATABOT_TOKEN)