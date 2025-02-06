import fastf1 as ff1
import numpy as np
import pandas as pd
from constants import F1DATABOT_DATA_CACHE
from unblock import to_thread

ff1.Cache.enable_cache(F1DATABOT_DATA_CACHE)

@to_thread
def get_session(gp_year: int, gp_race: str, gp_session: str,
                laps=True, telemetry=True, weather=False, messages=False):

    session = ff1.get_session(gp_year, gp_race, gp_session)
    session.load()

    return session

@to_thread
def get_multiple_sessions(gp_sessions):
    sessions = []

    for sesh in gp_sessions:
        session = get_session(sesh['gp_year'], sesh['gp_race'], sesh['gp_session'])
        sessions.append(session)

    return sessions 
    
@to_thread
def get_multiple_session_laps(gp_sessions):

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

    session = ff1.get_testing_session(year, number, session_number)
    session.load()

    return(session)