"""Discord bot commands for Formula 1 data visualization.

This module provides a collection of Discord slash commands that generate various
visualizations of Formula 1 data. It handles user interactions, data loading,
and plot generation for different types of F1 analysis.

The module includes commands for:
- Track maps and circuit layouts
- Weather data visualization
- Tyre strategy analysis
- Lap time comparisons
- Telemetry analysis
- Position changes and race gaps
- Qualifying session analysis
- Corner-by-corner analysis

Each command follows a similar pattern:
1. Validates user input
2. Loads required session data
3. Generates or retrieves cached plots
4. Sends the visualization to the Discord channel

Dependencies:
    - discord.py: For Discord bot functionality
    - fastf1: For Formula 1 data access
    - plot_functions: For visualization generation
    - data: For session data loading
"""

from __future__ import annotations

import discord
import data
import plot_functions as pf
from constants import F1DATABOT_PLOT_CACHE
from classes import DiscordBot
from classes import ValidationException
from discord.ext import commands
from discord import app_commands
from discord import Interaction
from os.path import isfile
from fastf1.core import DataNotLoadedError


class PlotCommands(commands.Cog):
    """A Discord cog containing commands for generating Formula 1 data visualizations.

    This cog provides a collection of Discord slash commands that generate various
    plots and visualizations of Formula 1 data, including track maps, weather data,
    tyre strategies, and telemetry comparisons.

    Attributes:
        bot (DiscordBot): The Discord bot instance this cog is attached to.
    """

    def __init__(self, bot: DiscordBot) -> None:
        """Initialize the PlotCommands cog.

        Args:
            bot (DiscordBot): The Discord bot instance to attach this cog to.
        """
        self.bot=bot

    async def cog_app_command_error(self, 
                                    interaction: Interaction[discord.Client], 
                                    error: app_commands.AppCommandError) -> None:
        """Handle errors that occur during command execution.

        Provides user-friendly error messages and logs errors for debugging.
        Handles specific error types:
        - DataNotLoadedError: When requested session data is unavailable
        - ValidationException: When input validation fails
        - Other errors: Generic error message with logging

        Args:
            interaction (Interaction[discord.Client]): The interaction that caused the error.
            error (app_commands.AppCommandError): The error that occurred.

        Returns:
            None: Sends error messages to the user and logs the error.
        """
        
        webhook=interaction.followup
        mention = interaction.user.mention
        cname = interaction.command.name
        base_response = f'{mention} you requested {cname}'
        
        if isinstance(error.original, DataNotLoadedError):
            await webhook.send(content=f'{base_response}, but the data failed to load. The session you requested may not exist (yet).',
                               ephemeral=True)
        elif isinstance(error.original, ValidationException):
            await webhook.send(content = str(error.original), ephemeral=True)
        else:
            await webhook.send(content="An error occurred while processing the command, my creator has been notified.",
                               ephemeral=True)

        chann = self.bot.get_channel(1214881440190300191)
        await chann.send(error)
        self.bot.logger.exception(error)

        return await super().cog_app_command_error(interaction, error)
    

    @app_commands.command(name='track_map', 
                          description='Plot of the track map')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend')
    async def _track_map(self, interaction: discord.Interaction, year: int, race: str):
        """Generate and send a track map visualization.

        Creates a plot showing the circuit layout with corner markers for the
        specified Grand Prix. The plot is cached and reused if already generated.

        Args:
            interaction (discord.Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the race weekend (e.g., 'Monaco', 'Silverstone').

        Returns:
            None: Sends the track map image to the Discord channel.
        """
        await interaction.response.defer()
        webhook = interaction.followup

        params = {'year': year, 'race': race}
        params = pf.check_plot_params(params)
        year, race = map(params.get, ('year', 'race'))

        session = await data.get_session(year, race, 'q')
        filename = f"{F1DATABOT_PLOT_CACHE}/track_map_{session.event['EventName']}_{session.event.year}.png"
        if not isfile(filename):
            pf.plot_track_layout(session, save=True)

        with open(filename, 'rb') as f:
            picture = discord.File(f)
            await webhook.send(content = f"Here it is, {interaction.user.mention}",
                            file=picture)


    @app_commands.command(name='weather', 
                          description='Plot of weather over the session')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., Q for qualifying or R for race)')
    async def _weather(self, interaction: discord.Interaction, year: int, race: str, sesh: str):
        """Generate and send a weather visualization for a session.

        Creates a plot showing weather conditions (temperature, humidity, wind speed)
        throughout the specified session. The plot is cached and reused if already generated.

        Args:
            interaction (discord.Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the race weekend.
            sesh (str): The session identifier (e.g., 'Q' for qualifying, 'R' for race).

        Returns:
            None: Sends the weather plot image to the Discord channel.
        """
        await interaction.response.defer()
        webhook = interaction.followup

        params = {'year': year, 'race': race, 'sesh': sesh}
        params = pf.check_plot_params(params)
        year, race, sesh = map(params.get, ('year', 'race', 'sesh'))

        session = await data.get_session(year, race, sesh, telemetry=False)
        filename = f"{F1DATABOT_PLOT_CACHE}/weather {session.event['EventName']} {session.event.year} {session.session_info['Name']}.png"

        if not isfile(filename):
            pf.plot_weather(session, save=True)
        
        with open(filename, 'rb') as f:
            picture = discord.File(f)
            await webhook.send(content = f"Here it is, {interaction.user.mention}",
                            file=picture)


    @app_commands.command(name='tyre_strategy', 
                          description='Plot the tyres used during the session for all drivers')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., Q for qualifying or R for race)')
    async def _tyre_strategy(self, interaction: discord.Interaction, year: int, race: str, sesh: str):
        """Generate and send a tyre strategy visualization.

        Creates a plot showing the tyre compounds used by each driver throughout
        the session, including stint lengths and compound types. The plot is cached
        and reused if already generated.

        Args:
            interaction (discord.Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the race weekend.
            sesh (str): The session identifier (e.g., 'Q' for qualifying, 'R' for race).

        Returns:
            None: Sends the tyre strategy plot image to the Discord channel.
        """
        await interaction.response.defer()
        webhook = interaction.followup

        params = {'year': year, 'race': race, 'sesh': sesh}
        params = pf.check_plot_params(params)
        year, race, sesh = map(params.get, ('year', 'race', 'sesh'))

        session = await data.get_session(year, race, sesh, telemetry=False)
        filename = f"{F1DATABOT_PLOT_CACHE}/tyre_strategy_{session.event['EventName']}_{session.event.year}_{session.event.get_session_name(sesh)}.png"

        if not isfile(filename):
            pf.plot_tyre_strategies(session, save=True)
        
        with open(filename, 'rb') as f:
            picture = discord.File(f)
            await webhook.send(content = f"Here it is, {interaction.user.mention}",
                            file=picture)


    @app_commands.command(name='fastest_laps', 
                          description='Plot the fastest laps of all drivers in the session')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., Q for qualifying or R for race)')
    async def _fastest_laps(self, interaction: discord.Interaction, year: int, race: str, sesh: str):
        """Generate and send a fastest laps visualization.

        Creates a plot comparing the fastest lap times of all drivers in the session,
        showing their relative performance. The plot is cached and reused if already generated.

        Args:
            interaction (discord.Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the race weekend.
            sesh (str): The session identifier (e.g., 'Q' for qualifying, 'R' for race).

        Returns:
            None: Sends the fastest laps plot image to the Discord channel.
        """
        await interaction.response.defer()
        webhook = interaction.followup

        params = {'year': year, 'race': race, 'sesh': sesh}
        params = pf.check_plot_params(params)
        year, race, sesh = map(params.get, ('year', 'race', 'sesh'))

        session = await data.get_session(year, race, sesh, telemetry=False)
        filename = f"{F1DATABOT_PLOT_CACHE}/fastest_laps {session.event['EventName']} {session.event.year} {session.session_info['Name']}.png"

        if not isfile(filename):
            pf.plot_fastest_laps(session, save=True)
        
        with open(filename, 'rb') as f:
            picture = discord.File(f)
            await webhook.send(content = f"Here it is, {interaction.user.mention}",
                            file=picture)


    @app_commands.command(name='qualifying_fastest_laps', 
                          description='Plot the fastest laps of all drivers in each of the three quali sessions')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session. Can only be Q or SQ')
    async def _qualifying_fastest_laps(self, interaction: discord.Interaction, year: int, race: str, sesh: str):
        """Generate and send a qualifying fastest laps visualization.

        Creates a plot comparing the fastest lap times of all drivers across Q1, Q2, and Q3
        qualifying sessions. Only works for qualifying sessions (Q or SQ).

        Args:
            interaction (discord.Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the race weekend.
            sesh (str): The session identifier (must be 'Q' or 'SQ').

        Returns:
            None: Sends the qualifying fastest laps plot image to the Discord channel.

        Raises:
            ValidationException: If the session is not a qualifying session.
        """
        await interaction.response.defer()
        webhook = interaction.followup

        if sesh.upper() not in ['Q', 'SQ']:
            raise ValidationException("This chart can only be produced for Q or SQ sessions")

        params = {'year': year, 'race': race, 'sesh': sesh}
        params = pf.check_plot_params(params)
        year, race, sesh = map(params.get, ('year', 'race', 'sesh'))

        session = await data.get_session(year, race, sesh, telemetry=False)
        filename = f"{F1DATABOT_PLOT_CACHE}/qualifying_fastest_laps {session.event['EventName']} {session.event.year} {session.session_info['Name']}.png"

        if not isfile(filename):
            pf.plot_qualifying_fastest_laps(session, save=True)
        
        with open(filename, 'rb') as f:
            picture = discord.File(f)
            await webhook.send(content = f"Here it is, {interaction.user.mention}",
                            file=picture)


    @app_commands.command(name='speed_versus_laptime',
                          description='Scatter plot of the top speed of each driver against their fastest laptime')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., Q for qualifying or R for race)')
    async def _speed_versus_laptime(self, interaction: discord.Interaction, year: int, race: str, sesh: str):
        """Generate and send a speed vs laptime correlation plot.

        Creates a scatter plot comparing each driver's top speed against their fastest
        lap time, showing the relationship between straight-line speed and overall pace.

        Args:
            interaction (discord.Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the race weekend.
            sesh (str): The session identifier (e.g., 'Q' for qualifying, 'R' for race).

        Returns:
            None: Sends the speed vs laptime plot image to the Discord channel.
        """
        await interaction.response.defer()
        webhook = interaction.followup

        params = {'year': year, 'race': race, 'sesh': sesh}
        params = pf.check_plot_params(params)
        year, race, sesh = map(params.get, ('year', 'race', 'sesh'))

        session = await data.get_session(year, race, sesh)
        filename = f"{F1DATABOT_PLOT_CACHE}/speed_versus_laptime {session.event['EventName']} {session.event.year} {session.session_info['Name']}.png"

        if not isfile(filename):
            pf.plot_speed_versus_laptime(session, trap_loc='max', save=True)

        with open(filename, 'rb') as f:
            picture = discord.File(f)
            await webhook.send(content = f"Here it is, {interaction.user.mention}",
                            file=picture)
            

    @app_commands.command(name='compare_telemetry',
                          description='Plots speed, throttle, brake, DRS and laptime delta for two drivers')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., Q for qualifying or R for race)',
                        driver1='The first driver to compare',
                        driver2='The second driver to compare',
                        lap='Optionally provide the lap number')
    async def _compare_telemetry(self, interaction: discord.Interaction, year: int, race: str, sesh: str, driver1: str, driver2: str, lap:(int|None)=None):
        """Generate and send a telemetry comparison visualization.

        Creates a detailed telemetry comparison between two drivers, showing:
        - Speed traces
        - Throttle application
        - Brake application
        - DRS usage
        - Lap time delta

        Args:
            interaction (discord.Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the race weekend.
            sesh (str): The session identifier (e.g., 'Q' for qualifying, 'R' for race).
            driver1 (str): The three-letter code for the first driver (e.g., 'VER').
            driver2 (str): The three-letter code for the second driver (e.g., 'HAM').
            lap (int|None, optional): Specific lap number to compare. If None, uses fastest laps.

        Returns:
            None: Sends the telemetry comparison plot image to the Discord channel.
        """
        await interaction.response.defer()
        webhook = interaction.followup

        params = {'year': year, 'race': race, 'sesh': sesh, 'driver1': driver1, 'driver2': driver2, 'lap': lap}
        params = pf.check_plot_params(params)
        year, race, sesh, driver1, driver2, lap = map(params.get, ('year', 'race', 'sesh', 'driver1', 'driver2', 'lap'))

        session = await data.get_session(year, race, sesh)

        if lap == None:
            lap_text = "Fastest Lap"
        else:
            lap_text = f"Lap {lap}"
        filename = f"{F1DATABOT_PLOT_CACHE}/compare_telemetry {session.event['EventName']} {session.event.year} {session.session_info['Name']} {driver1} {driver2} {lap_text}.png"

        if not isfile(filename):
            pf.plot_telemetry_comparison(session, driver1, driver2, lap=lap, save=True)

        with open(filename, 'rb') as f:
            picture = discord.File(f)
            await webhook.send(content = f"Here it is, {interaction.user.mention}",
                            file=picture)
            

    @app_commands.command(name='sector_performance',
                          description='Plot the sector times for all drivers in each sector, ranked fastest to slowest')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., Q for qualifying or R for race)')
    async def _sector_performance(self, interaction: discord.Interaction, year: int, race: str, sesh: str):

        await interaction.response.defer()
        webhook = interaction.followup

        params = {'year': year, 'race': race, 'sesh': sesh}
        params = pf.check_plot_params(params)
        year, race, sesh = map(params.get, ('year', 'race', 'sesh'))

        session = await data.get_session(year, race, sesh, telemetry=False)
        filename = f"{F1DATABOT_PLOT_CACHE}/sector_performance {session.event['EventName']} {session.event.year} {session.session_info['Name']}.png"

        if not isfile(filename):
            pf.plot_sector_performance(session, save=True)

        with open(filename, 'rb') as f:
            picture = discord.File(f)
            await webhook.send(content = f"Here it is, {interaction.user.mention}",
                            file=picture)
            

    @app_commands.command(name='position_changes',
                          description='Plot the position of each driver on each lap of the session')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., Q for qualifying or R for race)',
                        highlight='Comma separated list of drivers to highlight')
    async def _position_changes(self, interaction: discord.Interaction, year: int, race: str, sesh: str, highlight: str = ''):

        if sesh.upper() not in ['R', 'S']:
            raise ValidationException("Position changes can only be charted for races or sprint races")

        await interaction.response.defer()
        webhook = interaction.followup

        params = {'year': year, 'race': race, 'sesh': sesh, 'highlight': highlight}
        params = pf.check_plot_params(params)
        year, race, sesh, highlight = map(params.get, ('year', 'race', 'sesh', 'highlight'))

        session = await data.get_session(year, race, sesh, telemetry=False)

        if highlight == '':
            hightlight_text = ''
        else:
            hightlight_text = f' {str.replace(highlight, ",", "_")}'

        filename = f"{F1DATABOT_PLOT_CACHE}/position_changes {session.event['EventName']} {session.event.year} {session.session_info['Name']}{hightlight_text}.png"

        if not isfile(filename):
            await pf.plot_position_changes(session, highlight=highlight, save=True)

        with open(filename, 'rb') as f:
            picture = discord.File(f)
            await webhook.send(content = f"Here it is, {interaction.user.mention}",
                            file=picture)
            

    @app_commands.command(name='ideal_laptimes',
                          description='Plot the actual and potential best laptime for each driver')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., Q for qualifying or R for race)')
    async def _ideal_laptimes(self, interaction: discord.Interaction, year: int, race: str, sesh: str):

        await interaction.response.defer()
        webhook = interaction.followup

        params = {'year': year, 'race': race, 'sesh': sesh}
        params = pf.check_plot_params(params)
        year, race, sesh = map(params.get, ('year', 'race', 'sesh'))

        session = await data.get_session(year, race, sesh, telemetry=False)
        filename = f"{F1DATABOT_PLOT_CACHE}/ideal_laptimes {session.event['EventName']} {session.event.year} {session.session_info['Name']}.png"

        if not isfile(filename):
            pf.plot_actual_vs_ideal_laptimes(session, save=True)

        with open(filename, 'rb') as f:
            picture = discord.File(f)
            await webhook.send(content = f"Here it is, {interaction.user.mention}",
                            file=picture)               


    @app_commands.command(name='corner_analysis',
                          description='Compare telemetry for two drivers on a specific corner')
    @app_commands.describe(year='The year of the session',
                           race='Name of the race weekend',
                           sesh='The session (e.g., Q for qualifying or R for race)',
                           driver1='The three letter code for the first driver e.g., VER',
                           driver2='The three letter code for the second driver e.g., LEC',
                           corner='The number of the corner to analyse',
                           upper_offset='The length in metres to extend the analysis past the corner apex',
                           lower_offset='The length in metres to extend the analysis prior to the apex',
                           lap_number='The lap number, or leave blank to use the fastest lap for each driver')
    async def _corner_analysis(self, interaction: discord.Interaction,
                              year:int, race:str, sesh:str, driver1:str, driver2:str,
                              corner:int, upper_offset:int=200, lower_offset:int=200,
                              lap_number:int=None):

        await interaction.response.defer()
        webhook = interaction.followup

        params = {'year': year, 'race': race, 'sesh': sesh, 'driver1': driver1,
                  'driver2': driver2, 'corner': corner, 'upper_offset': upper_offset,
                  'lower_offset': lower_offset, 'lap_number': lap_number}
        params = pf.check_plot_params(params)
        year, race, sesh, driver1, driver2, corner, upper_offset, lower_offset, lap_number = map(params.get, 
                        ('year', 'race', 'sesh', 'driver1', 'driver2', 'corner', 'upper_offset', 'lower_offset', 'lap_number'))

        if lap_number == None:
            lap_text = 'Fastest Lap'
        else:
            lap_text = f'Lap Number {lap_number}'

        session = await data.get_session(year, race, sesh)
        filename = f"{F1DATABOT_PLOT_CACHE}/Corner analysis {session.event['EventName']} {session.event.year} {session.session_info['Name']} {driver1} {driver2} {lap_text} Corner {corner}.png"                        

        if not isfile(filename):
            pf.plot_corner_analysis(session, driver1, driver2, corner, lower_offset=lower_offset,
                                     upper_offset=upper_offset, lap_number=lap_number,
                                       save=True)

        with open(filename, 'rb') as f:
            picture = discord.File(f)
            await webhook.send(content = f"Here it is, {interaction.user.mention}",
                            file=picture) 
            


    @app_commands.command(name='qualifying_laptime_evolution',
                          description='Plot the evolution of laptimes through the quali sessions')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session. This can only be Q or SQ.',
                        fastest_laps='Only plot fastest laps in each session (True) or all laps (False)')
    async def _qualifying_laptime_evolution(self, interaction: discord.Interaction, year: int, race: str, sesh: str, fastest_laps: bool = True):

        await interaction.response.defer()
        webhook = interaction.followup

        if sesh.upper() not in ['Q', 'SQ']:
            raise ValidationException("This chart can only be produced for Q or SQ sessions")

        params = {'year': year, 'race': race, 'sesh': sesh}
        params = pf.check_plot_params(params)
        year, race, sesh = map(params.get, ('year', 'race', 'sesh'))

        if (fastest_laps == True):
            lap_text = "Fastest Laps"
        else: 
            lap_text = "All laps"

        session = await data.get_session(year, race, sesh)
        filename = f"{F1DATABOT_PLOT_CACHE}/qualifying_lap_evolution {lap_text} {session.event['EventName']} {session.event.year} {session.session_info['Name']}.png"

        if not isfile(filename):
            pf.plot_qualifying_lap_evolution(session, fastest_laps, save=True)

        with open(filename, 'rb') as f:
            picture = discord.File(f)
            await webhook.send(content = f"Here it is, {interaction.user.mention}",
                            file=picture)    
            


    @app_commands.command(name='race_gaps',
                          description='Plot the cumulative race gaps between drivers')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session. This can only be R or S.',
                        highlight='Comma separated list of drivers to highlight')
    async def _race_gaps(self, interaction: discord.Interaction, year: int, race: str, sesh: str, highlight: str = ''):

        await interaction.response.defer()
        webhook = interaction.followup

        if sesh.upper() not in ['R', 'S']:
            raise ValidationException("This chart can only be produced for R or S sessions")

        params = {'year': year, 'race': race, 'sesh': sesh, 'highlight': highlight}
        params = pf.check_plot_params(params)
        year, race, sesh, highlight = map(params.get, ('year', 'race', 'sesh', 'highlight'))

        session = await data.get_session(year, race, sesh)

        if highlight == '':
            hightlight_text = ''
        else:
            hightlight_text = f' {str.replace(highlight, ",", "_")}'

        filename = f"{F1DATABOT_PLOT_CACHE}/race_gaps {session.event['EventName']} {session.event.year} {session.session_info['Name']}{hightlight_text}.png"

        if not isfile(filename):
            pf.plot_race_gaps(session, save=True, highlight=highlight)

        with open(filename, 'rb') as f:
            picture = discord.File(f)
            await webhook.send(content = f"Here it is, {interaction.user.mention}",
                            file=picture)  
