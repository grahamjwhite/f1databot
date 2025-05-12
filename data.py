import fastf1 as ff1
import numpy as np
import pandas as pd
from constants import F1DATABOT_DATA_CACHE
from unblock import to_thread

ff1.Cache.enable_cache(F1DATABOT_DATA_CACHE)

@to_thread
def get_session(gp_year: int, gp_race: str, gp_session: str,
                laps=True, telemetry=True, weather=False, messages=False):
    """Load a Formula 1 session from the FastF1 cache.

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
    session = ff1.get_session(gp_year, gp_race, gp_session)
    session.load()

    return session

@to_thread
def get_multiple_sessions(gp_sessions):
    """Load multiple Formula 1 sessions from the FastF1 cache.

    Retrieves and loads data for multiple sessions specified in a list of
    session parameters.

    Args:
        gp_sessions (list): List of dictionaries containing session parameters.
            Each dictionary should have keys:
            - gp_year (int): Year of the Grand Prix
            - gp_race (str): Name of the Grand Prix
            - gp_session (str): Session identifier

    Returns:
        list: List of FastF1 Session objects containing the requested session data.
    """
    sessions = []

    for sesh in gp_sessions:
        session = get_session(sesh['gp_year'], sesh['gp_race'], sesh['gp_session'])
        sessions.append(session)

    return sessions 
    
@to_thread
def get_multiple_session_laps(gp_sessions):
    """Load and combine lap data from multiple Formula 1 sessions.

    Retrieves lap data from multiple sessions and combines them into a single
    DataFrame, adding columns to identify the year, race, and session for each lap.

    Args:
        gp_sessions (list): List of dictionaries containing session parameters.
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
    sessions = []
    for sesh in gp_sessions:
        session = get_session(sesh['gp_year'], sesh['gp_race'], sesh['gp_session'])
        
        laps = session.laps
        n = len(laps)
        laps['Year'] = sesh['gp_year'] * n
        laps['Race'] = sesh['gp_race'] * n
        laps['Session'] = sesh['gp_session'] * n

        sessions.append(laps)

    return pd.concat(sessions)
 
@to_thread
def get_test_session(year: int, number: int, session_number: int):
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
    session = ff1.get_testing_session(year, number, session_number)
    session.load()

    return(session)