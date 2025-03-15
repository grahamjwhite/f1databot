import numpy as np
import pandas as pd
import fastf1.plotting as ff1plt
import data
from fastf1.core import Laps, Session, Lap
from matplotlib.pyplot import axis
import f1_website as f1_web

def rotate(xy, *, angle):
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)


def fastest_laps_in_session(laps: Laps) -> Laps:

    # get a list of all drivers
    drivers = pd.unique(laps['Driver'])

    # get the fastest lap for each driver 
    list_fastest_laps = list()
    for drv in drivers:
        drvs_fastest_lap = laps.pick_drivers(drv).pick_fastest()
        if drvs_fastest_lap is not None: 
            list_fastest_laps.append(drvs_fastest_lap)
    fastest_laps = Laps(list_fastest_laps).sort_values(by='LapTime').reset_index(drop=True)

    # calculate percent difference from fastest lap
    pole_lap = fastest_laps.pick_fastest()
    fastest_laps['LapTimeDelta'] = (fastest_laps['LapTime'] - pole_lap['LapTime'])/pole_lap['LapTime']*100

    # remove any NA values
    fastest_laps = fastest_laps.dropna(subset="Driver") 

    return fastest_laps


def fastest_sectors_in_session(laps: Laps) -> pd.DataFrame:

    # get a list of all drivers
    drivers = pd.unique(laps['Driver'])

    # get the fastest sectors for each driver 
    sector1_times = []
    sector2_times = []
    sector3_times = []

    for drv in drivers:
        driver_laps = laps.pick_drivers(drv)
        sector1_times.append(driver_laps['Sector1Time'].min())
        sector2_times.append(driver_laps['Sector2Time'].min())
        sector3_times.append(driver_laps['Sector3Time'].min())

    times = pd.DataFrame({'driver': drivers,
                          'sector1': sector1_times, 
                          'sector2': sector2_times, 
                          'sector3': sector3_times})

    return times


def safety_car_laps(laps: Laps) -> Laps:
    
    sc_laps = laps[laps['TrackStatus'].str.contains('4', regex=False, na=False)]

    return sc_laps

def red_flag_laps(laps: Laps) -> Laps:

    rf_laps = laps[laps['TrackStatus'].str.contains('5', regex=False, na=False)]

    return rf_laps

def virtual_safety_car_laps(laps: Laps) -> Laps:

    vsc_laps = laps[laps['TrackStatus'].str.contains('|'.join('67'), regex=True, na=False)]

    return vsc_laps

def clean_laps(laps: Laps) -> Laps:

    green_laps = laps[~laps['TrackStatus'].str.contains('|'.join('4567'), regex=True, na=False)]

    return green_laps


def add_special_lap_shading(session: Session, ax: axis) -> None:

    # get the laps of the race leader
    lead_laps = session.laps.loc[session.laps["Position"]==1.0, :]

    # get the lap numbers of red flag, safety cars and vsc's
    rf_laps = np.unique(red_flag_laps(lead_laps)['LapNumber'].astype('int'))
    sc_laps = np.unique(safety_car_laps(lead_laps)['LapNumber'].astype('int'))
    vsc_laps = np.unique(virtual_safety_car_laps(lead_laps)['LapNumber'].astype('int'))

    # remove vsc laps that turned into a full safety car
    vsc_laps = vsc_laps[~np.isin(vsc_laps, sc_laps)]

    # draw a vertical red line for red flags
    if len(rf_laps) > 0:
        for rf_lap in rf_laps.tolist():
            ax.axvline(x=rf_lap, color = "red", alpha=1)

    # shade safety car laps orange
    if len(sc_laps) > 0:
        for sc_lap in sc_laps.tolist():
            ax.axvspan(xmin=sc_lap, xmax=sc_lap+0.95, color = "orange", alpha=0.2)

    # shade virtual safety car laps yellow
    if len(vsc_laps) > 0:
        for vsc_lap in vsc_laps.tolist():
            ax.axvspan(xmin=vsc_lap, xmax=vsc_lap+0.95, color = "yellow", alpha=0.2)


def fuel_corrected_times(stint_laps: Laps, 
                         laps_in_race: int, 
                         total_fuel: int = 100, 
                         time_per_kg: float = 0.03, 
                         reference_lap: int|None = None, 
                         return_value: str = 'timedelta'):
    
    laptimes = np.array(stint_laps['LapTime'].dt.total_seconds())
    lapnumbers = np.array(stint_laps['LapNumber'])

    # default rule-of-thumb: 0.03 seconds per kg of fuel per lap
    correction_per_lap = total_fuel/laps_in_race*time_per_kg

    # this will return the lap delta between each lap and the first
    # lap in the stint
    if return_value == 'timedelta':
        corrected_times = laptimes - laptimes[0] + correction_per_lap*np.arange(0, len(stint_laps))

    # this will return the actual laptimes, adjusted to the reference lap
    elif return_value == 'laptime':
        # if no reference lap number given, use the first lap in the stint
        if reference_lap == None:
            reference_lap = lapnumbers[0]

        corrected_times = laptimes - correction_per_lap*(reference_lap-lapnumbers)

    else:
        raise Exception("Unrecognised option for return_value. Use 'timedelta' or 'laptime'")

    return corrected_times


def fastest_drivers_in_team(session):
    teams = np.unique(session.laps['Team'])
    fastest_drivers = [session.laps.pick_team(team).pick_fastest()['Driver'] for team in teams]

    return fastest_drivers


def get_compound_color(compound_name, session):

    if compound_name == "TEST_UNKNOWN":
        color="#434649"
    else:
        color=ff1plt.get_compound_color(compound_name, session)

    return color



def finishing_order(session: Session) -> list:

    session_name = session.session_info['Name']

    if session_name in ["Race", "Sprint"]:
        driver_finish_times = session.laps[["Driver", "Time", "LapNumber"]].groupby("Driver", as_index=False).max()
        driver_finish_times = driver_finish_times.sort_values(by=["LapNumber", "Time"], ascending=[False, True])
        drivers_ordered = driver_finish_times["Driver"].tolist()

    elif session_name in ["Qualifying", "Sprint Qualifying"]:
        q1, q2, q3 = session.laps.pick_quicklaps().split_qualifying_sessions()
        q1 = fastest_laps_in_session(q1)
        q2 = fastest_laps_in_session(q2)
        q3 = fastest_laps_in_session(q3)

        top = q3["Driver"].tolist()
        middle = q2.loc[10:15,"Driver"].tolist()
        bottom = q1.loc[16:len(q1.index), "Driver"].tolist()
        drivers_ordered = top + middle + bottom

    else: # it's a prac session
        drivers_ordered = fastest_laps_in_session(session.laps).sort_values(by="LapTime", ascending=True)["Driver"].tolist()

    return drivers_ordered


def fix_positions(laps: Laps):

    # Where cars stop: set 'Position' to NaN; set 'Time' to NaT 
    laps.loc[laps["FastF1Generated"], "Position"] = np.NaN
    laps.loc[laps["FastF1Generated"], "Time"] = pd.NaT

    # Re-calculate the correct positions
    # Code taken from FastF1's core.py
    for lap_n in laps['LapNumber'].unique():
    # get each drivers lap for the current lap number, sorted by
    # the time when each lap was set
        laps_eq_n = laps.loc[
            laps['LapNumber'] == lap_n, ('Time', 'Position')
        ].reset_index(drop=True).sort_values(by='Time')

        # number positions and restore previous order by index
        laps_eq_n['Position'] = range(1, len(laps_eq_n) + 1)
        laps.loc[laps['LapNumber'] == lap_n, 'Position'] \
            = laps_eq_n.sort_index()['Position'].to_list()

    return laps


async def fix_grid_positions(session: Session) -> Session:

    drivers = pd.unique(session.laps['Driver'])
    grid = f1_web.process_starting_grid(f1_web.get_session_results(session, "starting-grid"))

    # get drivers with no grid position
    zero_grid_pos_drivers = drivers[~pd.Series(drivers).isin(grid["Abbreviation"])]

    # if there are no drivers to fix, return the session object unchanged
    if zero_grid_pos_drivers.size == 0:
        return session

    # get their qualifying order
    year = int(session.event.year)
    race = session.event['EventName']
    name = session.session_info['Name']

    if name == "Race":
        quali_session = await data.get_session(year, race, "Q")
    elif name == "Sprint Race":
        quali_session = await data.get_session(year, race, "SQ")
    
    # calculate their new position at the back of the grid
    quali_results = quali_session.results
    select_list = [x in zero_grid_pos_drivers.tolist() for x in quali_session.results["Abbreviation"]]
    quali_results = quali_results[np.array(select_list).astype('bool')].copy()
    last_nonzero_pos = len(drivers) - len(zero_grid_pos_drivers)
    quali_results["NewGridPosition"] = np.arange(1, len(zero_grid_pos_drivers)+1) + last_nonzero_pos

    # write it to the session object
    for _, driver in quali_results.iterrows():
        session.results.loc[session.results["Abbreviation"] == driver['Abbreviation'], "GridPosition"] = driver['NewGridPosition']

    return session
