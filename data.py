"""Data loading functions for Formula 1 sessions using FastF1.

This module provides asynchronous functions to fetch and load Formula 1 session
data using the FastF1 library. It handles data caching and allows loading
of various data types (laps, telemetry, weather, messages).

The module provides functions for:
- Loading single sessions
- Loading multiple sessions in parallel
- Loading and combining lap data from multiple sessions
- Loading testing sessions

Functions use the `@to_thread` decorator to run FastF1's synchronous loading
functions in separate threads, preventing blocking of the main application.

Cache configuration is handled via the F1DATABOT_DATA_CACHE constant.

Dependencies:
    - fastf1: For accessing Formula 1 data
    - pandas: For data manipulation (used in get_multiple_session_laps)
    - numpy: Required by FastF1/Pandas
    - constants: For cache path configuration
    - unblock: For the @to_thread decorator
"""

import fastf1 as ff1
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from fastf1.core import Session
from constants import F1DATABOT_DATA_CACHE
from unblock import to_thread

ff1.Cache.enable_cache(F1DATABOT_DATA_CACHE)

@to_thread
def get_session(gp_year: int, 
                gp_race: str, 
                gp_session: str,
                laps: bool = True, 
                telemetry: bool = True, 
                weather: bool = False, 
                messages: bool = False) -> Session:
    """Load a Formula 1 session from using FastF1.

    Retrieves and loads session data from the FastF1 cache, including lap times,
    telemetry, weather data, and team radio messages as specified.

    Args:
        gp_year (int): Year of the Grand Prix.
        gp_race (str): Name of the Grand Prix (e.g., 'Monaco', 'Silverstone').
        gp_session (str): Session identifier ('FP1', 'FP2', 'FP3', 'Q', 'SQ', 'R', 'S').
        laps (bool, optional): Whether to load lap timing data. Defaults to True.
        telemetry (bool, optional): Whether to load telemetry data. Defaults to True.
        weather (bool, optional): Whether to load weather data. Defaults to False.
        messages (bool, optional): Whether to load team radio messages. Defaults to False.

    Returns:
        Session: FastF1 Session object containing the requested session data.
    """
    session: Session = ff1.get_session(gp_year, gp_race, gp_session)
    session.load(laps=laps, telemetry=telemetry, weather=weather, messages=messages)

    return session

@to_thread
def get_multiple_sessions(gp_sessions: List[Dict[str, Any]]) -> List[Session]:
    """Load multiple Formula 1 sessions using FastF1.

    Retrieves and loads data for multiple sessions specified in a list of
    session parameters.

    Args:
        gp_sessions (List[Dict[str, Any]]): List of dictionaries containing session parameters.
            Each dictionary should have keys:
            - gp_year (int): Year of the Grand Prix
            - gp_race (str): Name of the Grand Prix
            - gp_session (str): Session identifier

    Returns:
        List[Session]: List of FastF1 Session objects containing the requested session data.
    """
    sessions: List[Session] = []

    for sesh in gp_sessions:
        session: Session = get_session(sesh['gp_year'], sesh['gp_race'], sesh['gp_session'])
        sessions.append(session)

    return sessions 
    
@to_thread
def get_multiple_session_laps(gp_sessions: List[Dict[str, Any]]) -> pd.DataFrame:
    """Load and combine lap data from multiple Formula 1 sessions.

    Retrieves lap data from multiple sessions and combines them into a single
    DataFrame, adding columns to identify the year, race, and session for each lap.

    Args:
        gp_sessions (List[Dict[str, Any]]): List of dictionaries containing session parameters.
            Each dictionary should have keys:
            - gp_year (int): Year of the Grand Prix
            - gp_race (str): Name of the Grand Prix
            - gp_session (str): Session identifier

    Returns:
        pd.DataFrame: Combined DataFrame containing lap data from all sessions,
            with additional columns:
            - Year: Year of the Grand Prix
            - Race: Name of the Grand Prix
            - Session: Session identifier
    """
    laps_list: List[pd.DataFrame] = []
    for sesh in gp_sessions:
        session: Session = get_session(sesh['gp_year'], sesh['gp_race'], sesh['gp_session'])
        
        laps: pd.DataFrame = session.laps
        n: int = len(laps)
        laps['Year'] = sesh['gp_year']
        laps['Race'] = sesh['gp_race']
        laps['Session'] = sesh['gp_session']

        laps_list.append(laps)

    if not laps_list:
        return pd.DataFrame() 
        
    return pd.concat(laps_list)
 
@to_thread
def get_test_session(year: int, number: int, session_number: int) -> Session:
    """Load a Formula 1 testing session from the FastF1 cache.

    Retrieves and loads data for a specific testing session, including pre-season
    and in-season testing events.

    Args:
        year (int): Year of the testing session.
        number (int): Test event number (1 for pre-season, 2 for in-season).
        session_number (int): Session number within the test event.

    Returns:
        Session: FastF1 Session object containing the testing session data.
    """
    session: Session = ff1.get_testing_session(year, number, session_number)
    session.load()

    return session