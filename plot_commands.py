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
    - plot_functions: For visualization functions
    - data: For session data loading
    - classes: For custom bot class and exceptions
    - constants: For configuration
    - asyncio: For asynchronous operations
    - logging: For error logging
    - functools: For wrapping plot functions
    - typing: For type hints
"""

import discord
import fastf1 as ff1 # Renamed for consistency
import plot_functions as pf
import data
import logging
import asyncio
import functools
import inspect # Used for signature inspection

from discord import app_commands, Interaction, File
from discord.ext import commands
from fastf1.core import Session
from classes import DiscordBot, ValidationException
from constants import F1DATABOT_PLOT_CACHE, F1DATABOT_ERR_CHANNEL
from typing import Callable, Any, Coroutine, Dict, Optional, Tuple
from fastf1._api import SessionNotAvailableError
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
        self.bot: DiscordBot = bot

    async def cog_app_command_error(self, 
                                    interaction: Interaction, 
                                    error: app_commands.AppCommandError) -> None:
        """Handles errors raised during application command execution.

        Logs the error and sends an appropriate message to the user or error channel.

        Args:
            interaction (Interaction): The interaction that caused the error.
            error (app_commands.AppCommandError): The error that was raised.
        """
        # Extract original error if it exists
        original_error: BaseException = getattr(error, 'original', error)
        
        error_message: str = f"Error executing command '{interaction.command.name if interaction.command else 'unknown'}': {original_error}"
        self.bot.logger.error(error_message, exc_info=original_error)

        user_message: str = "An unexpected error occurred. Please try again later."
        if isinstance(original_error, ValidationException):
            user_message = str(original_error)
        elif isinstance(original_error, SessionNotAvailableError):
            user_message = "Session data not available, it might be too old or not exist for this year/race/session."
        elif isinstance(original_error, DataNotLoadedError):
            user_message = "Required data could not be loaded for the session."
        elif isinstance(original_error, TimeoutError): # Check for specific timeout
            user_message = "The request timed out while loading data. Please try again."
        # Add more specific FastF1/data loading errors if needed

        try:
            if interaction.response.is_done():
                await interaction.followup.send(user_message, ephemeral=True)
            else:
                await interaction.response.send_message(user_message, ephemeral=True)
        except discord.errors.NotFound:
            # Interaction might have expired, log and potentially send to error channel
            self.bot.logger.warning(f"Interaction expired before error message could be sent for command: {interaction.command.name if interaction.command else 'unknown'}")
            # Optionally send detailed error to a specific channel
            try:
                err_channel_id: int = F1DATABOT_ERR_CHANNEL
                err_channel: Optional[discord.abc.Messageable] = self.bot.get_channel(err_channel_id)
                if err_channel:
                    await err_channel.send(f"Error in interaction (expired): {error_message}\nDetails: {original_error}")
            except Exception as ch_err:
                 self.bot.logger.error(f"Failed to send error to error channel: {ch_err}")
        except Exception as e:
            # Catch other potential errors during response/followup
            self.bot.logger.error(f"Error sending error message to user: {e}", exc_info=e)

    async def _plot_template(self, interaction: Interaction, year: int, race: str, sesh: str, 
                             plot_func: Callable[..., Coroutine[Any, Any, Optional[str]]],
                             load_laps: bool = True, 
                             load_telemetry: bool = True, 
                             load_weather: bool = False, 
                             load_messages: bool = False,
                             **kwargs: Any) -> None:
        """Template function for handling plot generation commands.

        Handles data loading, plot generation (calling the specific plot_func),
        error handling, and sending the plot file to Discord.

        Args:
            interaction (Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the Grand Prix.
            sesh (str): The session identifier ('FP1', 'Q', 'R', etc.).
            plot_func (Callable[..., Coroutine[Any, Any, Optional[str]]]): The async plotting function to call.
                Should return the filename stem if saved, else None.
            load_laps (bool, optional): Whether to load lap data. Defaults to True.
            load_telemetry (bool, optional): Whether to load telemetry data. Defaults to True.
            load_weather (bool, optional): Whether to load weather data. Defaults to False.
            load_messages (bool, optional): Whether to load session messages. Defaults to False.
            **kwargs: Additional keyword arguments specific to the plot function.
        """
        await interaction.response.defer(thinking=True)
        
        func_params: Dict[str, Any] = {'year': year, 'race': race, 'sesh': sesh, **kwargs}

        if not (2018 <= year <= 2025): 
            raise ValidationException("Invalid year provided.")
        if not race:
            raise ValidationException("Race name cannot be empty.")
        if not sesh:
             raise ValidationException("Session identifier cannot be empty.")

        checked_kwargs: Dict[str, Any] = pf.check_plot_params(kwargs)

        try:
            session: Session = await data.get_session(year, race, sesh, 
                                                      laps=load_laps, 
                                                      telemetry=load_telemetry, 
                                                      weather=load_weather, 
                                                      messages=load_messages)
        except SessionNotAvailableError as e:
            await interaction.followup.send(f"Session data not available: {e}", ephemeral=True)
            return
        except Exception as e:
            self.bot.logger.error(f"Error loading session {year} {race} {sesh}: {e}", exc_info=e)
            await interaction.followup.send("Failed to load session data. Please check inputs or try later.", ephemeral=True)
            return

        plot_args: Dict[str, Any] = {'session': session, **checked_kwargs}
        
        sig: inspect.Signature = inspect.signature(plot_func)
        valid_plot_args: Dict[str, Any] = {k: v for k, v in plot_args.items() if k in sig.parameters}

        try:
            saved_filename_stem: Optional[str] = await plot_func(**valid_plot_args)
            
            if saved_filename_stem:
                full_path: str = f"{F1DATABOT_PLOT_CACHE}/{saved_filename_stem}"
                await interaction.followup.send(file=File(full_path))
            else:
                # This case handles if plot_func returned None (e.g., plot not saved or error before saving)
                error_msg = f"Plot function '{plot_func.__name__}' did not return a filename. Plot might not have been saved or generation failed."
                self.bot.logger.warning(error_msg) # Log as warning, might not be a critical error for all cases
                await interaction.followup.send("Sorry, the plot could not be generated or was not saved.", ephemeral=True)

        except FileNotFoundError: # This might still be relevant if the file is expected but somehow missing after generation
            # Construct the expected filename for logging, even if it's from the new generator
            # This part is tricky as we don't *know* the filename if saved_filename_stem was None
            # For robustness, we log a generic message if saved_filename_stem was None here.
            expected_fn_for_log = f"{plot_func.__name__}_...png" # Placeholder
            if saved_filename_stem: # if we had a name, use it in log
                expected_fn_for_log = f"{F1DATABOT_PLOT_CACHE}/{saved_filename_stem}"
            
            error_msg = f"Plot file not found: {expected_fn_for_log}. Plot generation might have failed or the file was unexpectedly removed."
            self.bot.logger.error(error_msg)
            await interaction.followup.send("Sorry, the plot file was not found after generation.", ephemeral=True)
        except ValidationException as e:
            raise e
        except Exception as e:
            error_msg = f"Error generating/sending plot '{plot_func.__name__}' for {year} {race} {sesh}: {e}"
            self.bot.logger.error(error_msg, exc_info=e)
            raise app_commands.AppCommandError("An error occurred during plot generation.") from e

    @app_commands.command(name='track_map', 
                          description='Plot the track map of the circuit')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., Q for qualifying or R for race)')
    async def _track_map(self, interaction: Interaction, year: int, race: str, sesh: str) -> None:
        """Generate and send a track layout visualization.

        Calls the plot template with the plot_track_layout function.

        Args:
            interaction (Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the Grand Prix.
            sesh (str): The session identifier ('FP1', 'Q', 'R', etc.).
        """
        await self._plot_template(interaction, year, race, sesh, pf.plot_track_layout)

    @app_commands.command(name='weather', 
                          description='Plot the weather during the session')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., Q for qualifying or R for race)')
    async def _weather(self, interaction: Interaction, year: int, race: str, sesh: str) -> None:
        """Generate and send a weather conditions visualization.

        Calls the plot template with the plot_weather function.

        Args:
            interaction (Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the Grand Prix.
            sesh (str): The session identifier ('FP1', 'Q', 'R', etc.).
        """
        await self._plot_template(interaction, year, race, sesh, pf.plot_weather, load_weather=True)

    @app_commands.command(name='fastest_laps', 
                          description='Plot the fastest laps of all drivers in the session')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., Q for qualifying or R for race)')
    async def _fastest_laps(self, interaction: Interaction, year: int, race: str, sesh: str) -> None:
        """Generate and send a fastest laps visualization.

        Creates a plot comparing the fastest lap times of all drivers in the session,
        showing their relative performance. The plot is cached and reused if already generated.

        Args:
            interaction (Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the Grand Prix.
            sesh (str): The session identifier ('FP1', 'Q', 'R', etc.).
        """
        await self._plot_template(interaction, year, race, sesh, pf.plot_fastest_laps)

    @app_commands.command(name='qualifying_fastest_laps', 
                          description='Plot the lap times of all drivers across the qualifying session')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend')
    async def _qualifying_fastest_laps(self, interaction: Interaction, year: int, race: str) -> None:
        """Generate and send a qualifying lap times visualization.

        Creates a three-panel plot showing lap times for each driver in Q1, Q2, and Q3.

        Args:
            interaction (Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the Grand Prix.
        """
        await self._plot_template(interaction, year, race, 'Q', pf.plot_qualifying_fastest_laps)

    @app_commands.command(name='qualifying_laptime_evolution',
                          description='Plot the evolution of lap times during a qualifying session.')
    @app_commands.describe(year='The year of the session',
                           race='Name of the race weekend',
                           fastest_laps='Whether to show only fastest laps (True) or all laps (False). Defaults to True.')
    async def _qualifying_laptime_evolution(self, interaction: Interaction, year: int, race: str, fastest_laps: bool = True) -> None:
        """Generate and send a qualifying lap time evolution plot.

        Args:
            interaction (Interaction): The Discord interaction.
            year (int): The year of the Grand Prix.
            race (str): The name of the Grand Prix.
            fastest_laps (bool, optional): Show only fastest laps. Defaults to True.
        """
        await self._plot_template(interaction, year, race, 'Q',
                                  pf.plot_qualifying_laptime_evolution, fastest_laps=fastest_laps)

    @app_commands.command(name='tyre_strategy', 
                          description='Plot the tyre strategies of all drivers during the session')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., R for race or S for sprint)')
    async def _tyre_strategies(self, interaction: Interaction, year: int, race: str, sesh: str) -> None:
        """Generate and send a tyre strategy visualization.

        Creates a Gantt-style chart showing tyre compounds used by each driver
        throughout the session.

        Args:
            interaction (Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the Grand Prix.
            sesh (str): The session identifier ('R', 'S').
        """
        await self._plot_template(interaction, year, race, sesh, pf.plot_tyre_strategies)

    @app_commands.command(name='compare_telemetry', 
                          description='Compare the telemetry (speed, throttle, brake, gear, RPM) of two drivers')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., Q for qualifying or R for race)',
                        driver1='The 3-letter code of the first driver',
                        driver2='The 3-letter code of the second driver')
    async def _compare_telemetry(self, interaction: Interaction, year: int, race: str, sesh: str, driver1: str, driver2: str) -> None:
        """Generate and send a telemetry comparison plot for two drivers.

        Creates a multi-panel plot comparing Speed, Throttle, Brake, Gear, and RPM
        for the fastest laps of the two specified drivers.

        Args:
            interaction (Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the Grand Prix.
            sesh (str): The session identifier ('FP1', 'Q', 'R', etc.).
            driver1 (str): 3-letter code of the first driver.
            driver2 (str): 3-letter code of the second driver.
        """
        if driver1.upper() == driver2.upper():
            await interaction.response.send_message("Please select two different drivers for comparison.", ephemeral=True)
            return
        await self._plot_template(interaction, year, race, sesh, pf.plot_telemetry_comparison, driver1=driver1, driver2=driver2)

    @app_commands.command(name='corner_analysis', 
                          description='Compare the cornering telemetry (speed, throttle, brake) of two drivers')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., Q for qualifying or R for race)',
                        driver1='The 3-letter code of the first driver',
                        driver2='The 3-letter code of the second driver',
                        corner='The corner number to compare')
    async def _corner_analysis(self, interaction: Interaction, year: int, race: str, sesh: str, driver1: str, driver2: str, corner: int) -> None:
        """Generate and send a corner telemetry comparison plot.

        Compares Speed, Throttle, and Brake application for two drivers through
        a specific corner.

        Args:
            interaction (Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the Grand Prix.
            sesh (str): The session identifier ('FP1', 'Q', 'R', etc.).
            driver1 (str): 3-letter code of the first driver.
            driver2 (str): 3-letter code of the second driver.
            corner (int): The corner number to analyze.
        """
        if driver1.upper() == driver2.upper():
            await interaction.response.send_message("Please select two different drivers for comparison.", ephemeral=True)
            return
        await self._plot_template(interaction, year, race, sesh, pf.plot_corner_analysis, driver1=driver1, driver2=driver2, corner=corner)

    @app_commands.command(name='sector_performance', 
                          description='Compare the theoretical fastest laps of all drivers sector-by-sector')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., Q for qualifying or R for race)')
    async def _sector_performance(self, interaction: Interaction, year: int, race: str, sesh: str) -> None:
        """Generate and send a sector time comparison plot.

        Creates a stacked bar chart showing each driver's best individual sector
        times, representing their theoretical fastest lap.

        Args:
            interaction (Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the Grand Prix.
            sesh (str): The session identifier ('FP1', 'Q', 'R', etc.).
        """
        await self._plot_template(interaction, year, race, sesh, pf.plot_sector_performance)

    @app_commands.command(name='gears', 
                          description='Plot the gear selection around the track')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., Q for qualifying or R for race)')
    async def _gears(self, interaction: Interaction, year: int, race: str, sesh: str) -> None:
        """Generate and send a gear selection visualization on the track map.

        Creates a color-coded visualization of the track layout showing gear usage
        at different points during the fastest lap.

        Args:
            interaction (Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the Grand Prix.
            sesh (str): The session identifier ('FP1', 'Q', 'R', etc.).
        """
        await self._plot_template(interaction, year, race, sesh, pf.plot_gears_on_track)

    @app_commands.command(name='speed_versus_laptime', 
                          description='Plot speed trap vs lap time for all drivers')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., Q for qualifying or R for race)')
    async def _speed_versus_laptime(self, interaction: Interaction, year: int, race: str, sesh: str) -> None:
        """Generate and send a speed vs. lap time correlation plot.

        Creates a scatter plot comparing each driver's fastest lap time against their
        highest speed trap measurement.

        Args:
            interaction (Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the Grand Prix.
            sesh (str): The session identifier ('FP1', 'Q', 'R', etc.).
        """
        # Speed trap location might need to be determined or passed
        # For now, assume plot_speed_versus_laptime handles finding the trap if None
        await self._plot_template(interaction, year, race, sesh, pf.plot_speed_versus_laptime, trap_loc=None)

    @app_commands.command(name='actions', 
                          description='Plot throttle, brake and gear selection around the track')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., Q for qualifying or R for race)',
                        driver='The 3-letter code of the driver to analyze')
    async def _actions(self, interaction: Interaction, year: int, race: str, sesh: str, driver: str) -> None:
        """Generate and send a driver actions visualization on the track map.

        Creates a color-coded track map showing throttle application, brake usage,
        and gear selection during the fastest lap.

        Args:
            interaction (Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the Grand Prix.
            sesh (str): The session identifier ('FP1', 'Q', 'R', etc.).
            driver (str): The 3-letter code of the driver to analyze.
        """
        await self._plot_template(interaction, year, race, sesh, pf.plot_actions_on_track, driver=driver)

    @app_commands.command(name='ideal_laptimes', 
                          description='Compare fastest lap times against theoretical ideal lap times')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., Q for qualifying or R for race)')
    async def _ideal_laptimes(self, interaction: Interaction, year: int, race: str, sesh: str) -> None:
        """Generate and send an ideal vs. actual lap time comparison plot.

        Creates a two-panel plot showing the gap between each driver's actual fastest
        lap and their theoretical ideal lap (sum of best sectors).

        Args:
            interaction (Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the Grand Prix.
            sesh (str): The session identifier ('FP1', 'Q', 'R', etc.).
        """
        await self._plot_template(interaction, year, race, sesh, pf.plot_actual_vs_ideal_laptimes)

    @app_commands.command(name='position_changes', 
                          description='Plot the positions of the drivers throughout the session')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., R for race or S for sprint)',
                        highlight='Enter a driver code to highlight them on the chart')
    async def _race_positions(self, interaction: Interaction, year: int, race: str, sesh: str, highlight: str = '') -> None:
        """Generate and send a position change visualization.

        Creates a line plot showing the position of each driver on each lap
        throughout the session. Optionally highlights a specific driver.

        Args:
            interaction (Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the Grand Prix.
            sesh (str): The session identifier ('R', 'S').
            highlight (str, optional): 3-letter code of the driver to highlight. Defaults to ''.
        """
        await self._plot_template(interaction, year, race, sesh, pf.plot_position_changes, highlight=highlight)

    @app_commands.command(name='race_gaps', 
                          description='Plot the gaps between drivers during the session')
    @app_commands.describe(year='The year of the session', 
                        race='Name of the race weekend',
                        sesh='The session (e.g., R for race or S for sprint)',
                        highlight='Enter a driver code to highlight them on the chart')
    async def _race_gaps(self, interaction: Interaction, year: int, race: str, sesh: str, highlight: str = '') -> None:
        """Generate and send a race gaps visualization.

        Creates a plot showing the time gap between each driver and the race leader
        on each lap. Optionally highlights a specific driver.

        Args:
            interaction (Interaction): The Discord interaction that triggered the command.
            year (int): The year of the Grand Prix.
            race (str): The name of the Grand Prix.
            sesh (str): The session identifier ('R', 'S').
            highlight (str, optional): 3-letter code of the driver to highlight. Defaults to ''.
        """
        await self._plot_template(interaction, year, race, sesh, pf.plot_race_gaps, highlight=highlight)  