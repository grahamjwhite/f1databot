"""Utility functions for processing and analyzing Formula 1 data.

This module provides a collection of helper functions used throughout the application
for various data manipulation, analysis, and plotting support tasks related to
Formula 1 data obtained from FastF1 and other sources.

Key functionalities include:
- Geometric transformations (coordinate rotation)
- Lap and sector time analysis (finding fastest laps/sectors, ideal laps)
- Filtering laps based on track status (SC, VSC, Red Flags)
- Data enrichment (fuel correction, position fixing, grid position fixing)
- Plotting assistance (special condition shading, tyre colors)
- Determining session outcomes (finishing order)

These functions support the main data loading, plotting, and command handling
modules by providing reusable data processing logic.

Dependencies:
    - numpy: For numerical operations and array manipulation.
    - pandas: For data structures and manipulation (DataFrames, Series).
    - fastf1: For core F1 data types (Laps, Session) and plotting utilities.
    - matplotlib: For plotting elements (axis manipulation).
    - f1_website: For accessing starting grid data.
    - data: For accessing session data when fixing grid positions.
"""

import numpy as np
import pandas as pd
import fastf1.plotting as ff1plt
import data
from typing import List, Optional, Union, Any # Added Any for potential complex types
from fastf1.core import Laps, Lap, Session
from matplotlib.axes import Axes # Correct import for Axes type
import f1_website as f1_web

def rotate(xy: np.ndarray, *, angle: float) -> np.ndarray:
    """Rotate a set of 2D coordinates by a specified angle.

    Args:
        xy (np.ndarray): Array of shape (n, 2) containing x,y coordinates to rotate.
        angle (float): Rotation angle in radians.

    Returns:
        numpy.ndarray: Array of shape (n, 2) containing the rotated coordinates.
    """
    rot_mat: np.ndarray = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)


def fastest_laps_in_session(laps: Laps) -> Laps:
    """Find the fastest lap for each driver in a session and calculate time deltas.

    Processes lap data to:
    - Find the fastest lap for each driver
    - Sort laps by lap time
    - Calculate percentage difference from fastest lap
    - Remove any invalid entries

    Args:
        laps (Laps): FastF1 Laps object containing session lap data.

    Returns:
        Laps: FastF1 Laps object containing:
            - Fastest lap for each driver
            - LapTimeDelta column showing percentage difference from fastest lap
            - Sorted by lap time
    """

    drivers: np.ndarray = pd.unique(laps['Driver'])
    list_fastest_laps: list = list()
    for drv in drivers:
        drvs_fastest_lap: Lap = laps.pick_drivers(drv).pick_fastest() # pick_fastest can return Lap or None
        if drvs_fastest_lap is not None: 
            list_fastest_laps.append(drvs_fastest_lap)
    
    if not list_fastest_laps:
        return Laps() # Return empty Laps if no fastest laps found
        
    fastest_laps_df: Laps = Laps(list_fastest_laps).sort_values(by='LapTime').reset_index(drop=True)

    pole_lap: pd.Series = fastest_laps_df.pick_fastest()
    # Ensure pole_lap is not None before accessing LapTime
    if pole_lap is not None and not pd.isnull(pole_lap['LapTime']):
        fastest_laps_df['LapTimeDelta'] = (fastest_laps_df['LapTime'] - pole_lap['LapTime'])/pole_lap['LapTime']*100
    else:
        fastest_laps_df['LapTimeDelta'] = np.nan # Assign NaN if pole lap is invalid
        
    fastest_laps_df = fastest_laps_df.dropna(subset=["Driver"]) 
    return fastest_laps_df


def fastest_sectors_in_session(laps: Laps) -> pd.DataFrame:
    """Find the fastest sector times for each driver in a session.

    Processes lap data to find the minimum sector time achieved by each driver
    in each sector of the track.

    Args:
        laps (Laps): FastF1 Laps object containing session lap data.

    Returns:
        pd.DataFrame: DataFrame containing:
            - driver: Driver codes
            - sector1: Fastest sector 1 time for each driver
            - sector2: Fastest sector 2 time for each driver
            - sector3: Fastest sector 3 time for each driver
    """

    drivers: np.ndarray = pd.unique(laps['Driver'])
    sector1_times: list = []
    sector2_times: list = []
    sector3_times: list = []

    for drv in drivers:
        driver_laps: Laps = laps.pick_drivers(drv)
        sector1_times.append(driver_laps['Sector1Time'].min())
        sector2_times.append(driver_laps['Sector2Time'].min())
        sector3_times.append(driver_laps['Sector3Time'].min())

    times: pd.DataFrame = pd.DataFrame({'driver': drivers,
                          'sector1': sector1_times, 
                          'sector2': sector2_times, 
                          'sector3': sector3_times})

    return times


def safety_car_laps(laps: Laps) -> Laps:
    """Filter laps to find those run under safety car conditions.

    Args:
        laps (Laps): FastF1 Laps object containing session lap data.

    Returns:
        Laps: FastF1 Laps object containing only laps run under safety car conditions.
    """
    # Ensure TrackStatus column exists and handle potential NaNs in contains
    if 'TrackStatus' not in laps.columns:
        return Laps() # Return empty Laps if column missing
    sc_laps: Laps = laps.loc[laps['TrackStatus'].astype(str).str.contains('4', regex=False, na=False)]
    return sc_laps

def red_flag_laps(laps: Laps) -> Laps:
    """Filter laps to find those run under red flag conditions.

    Args:
        laps (Laps): FastF1 Laps object containing session lap data.

    Returns:
        Laps: FastF1 Laps object containing only laps run under red flag conditions.
    """
    if 'TrackStatus' not in laps.columns:
        return Laps()
    rf_laps: Laps = laps.loc[laps['TrackStatus'].astype(str).str.contains('5', regex=False, na=False)]
    return rf_laps

def virtual_safety_car_laps(laps: Laps) -> Laps:
    """Filter laps to find those run under virtual safety car conditions.

    Args:
        laps (Laps): FastF1 Laps object containing session lap data.

    Returns:
        Laps: FastF1 Laps object containing only laps run under virtual safety car conditions.
    """
    if 'TrackStatus' not in laps.columns:
        return Laps()
    vsc_laps: Laps = laps.loc[laps['TrackStatus'].astype(str).str.contains('6|7', regex=True, na=False)] # Simplified regex
    return vsc_laps

def clean_laps(laps: Laps) -> Laps:
    """Filter laps to find those run under normal racing conditions.

    Removes laps run under safety car, virtual safety car, or red flag conditions.

    Args:
        laps (Laps): FastF1 Laps object containing session lap data.

    Returns:
        Laps: FastF1 Laps object containing only laps run under normal racing conditions.
    """
    if 'TrackStatus' not in laps.columns:
        return laps # Return original if no status column
    # Corrected regex pattern to exclude 4, 5, 6, 7
    green_laps: Laps = laps.loc[~laps['TrackStatus'].astype(str).str.contains('[4567]', regex=True, na=False)]
    return green_laps


def add_special_lap_shading(session: Session, ax: Axes) -> None:
    """Add visual indicators for special race conditions to a plot.

    Adds vertical lines and shaded regions to indicate:
    - Red flags (red vertical lines)
    - Safety car periods (orange shading)
    - Virtual safety car periods (yellow shading)

    Args:
        session (Session): FastF1 Session object containing race data.
        ax (Axes): Matplotlib axis object to add the shading to.

    Returns:
        None: Modifies the provided axis object.
    """

    lead_laps: Laps = session.laps[session.laps['Position'] == 1]
    rf_laps: np.ndarray = np.unique(red_flag_laps(lead_laps)['LapNumber'].astype('int'))
    sc_laps: np.ndarray = np.unique(safety_car_laps(lead_laps)['LapNumber'].astype('int'))
    vsc_laps: np.ndarray = np.unique(virtual_safety_car_laps(lead_laps)['LapNumber'].astype('int'))
    vsc_laps = vsc_laps[~np.isin(vsc_laps, sc_laps)]

    if len(rf_laps) > 0:
        for rf_lap in rf_laps.tolist():
            ax.axvline(x=rf_lap, color="red", alpha=1)

    if len(sc_laps) > 0:
        for sc_lap in sc_laps.tolist():
            ax.axvspan(xmin=sc_lap, xmax=sc_lap + 0.95, color="orange", alpha=0.2)

    if len(vsc_laps) > 0:
        for vsc_lap in vsc_laps.tolist():
            ax.axvspan(xmin=vsc_lap, xmax=vsc_lap + 0.95, color="yellow", alpha=0.2)


def fuel_corrected_times(stint_laps: Laps, 
                         laps_in_race: int, 
                         total_fuel: int = 100, 
                         time_per_kg: float = 0.03, 
                         reference_lap: Optional[int] = None, 
                         return_value: str = 'timedelta') -> np.ndarray:
    """Calculate fuel-corrected lap times for a stint.

    Adjusts lap times to account for fuel burn-off effects, either as:
    - Time deltas relative to the first lap
    - Absolute lap times adjusted to a reference lap

    Args:
        stint_laps (Laps): FastF1 Laps object containing the stint data.
        laps_in_race (int): Total number of laps in the race.
        total_fuel (int, optional): Total fuel load in kg. Defaults to 100.
        time_per_kg (float, optional): Time penalty per kg of fuel per lap in seconds.
            Defaults to 0.03.
        reference_lap (Optional[int], optional): Lap number to adjust times to.
            If None, uses first lap of stint. Defaults to None.
        return_value (str, optional): Type of correction to return.
            Options: 'timedelta' or 'laptime'. Defaults to 'timedelta'.

    Returns:
        np.ndarray: Array of corrected lap times.

    Raises:
        Exception: If return_value is not 'timedelta' or 'laptime'.
        ValueError: If stint_laps is empty or contains invalid data.
    """
    if stint_laps.empty:
        raise ValueError("Input Laps object 'stint_laps' is empty.")
        
    try:
        laptimes: np.ndarray = stint_laps['LapTime'].dt.total_seconds().to_numpy()
        lapnumbers: np.ndarray = stint_laps['LapNumber'].to_numpy()
    except KeyError as e:
        raise ValueError(f"Input Laps object missing required column: {e}") from e
    except AttributeError as e:
        raise ValueError(f"Invalid data type in Laps object: {e}") from e
        
    if laps_in_race <= 0:
         raise ValueError("laps_in_race must be positive.")

    correction_per_lap: float = total_fuel / laps_in_race * time_per_kg

    if return_value == 'timedelta':
        # Ensure laptimes is not empty before accessing laptimes[0]
        if laptimes.size == 0:
            return np.array([]) # Return empty array if no laptimes
        corrected_times: np.ndarray = laptimes - laptimes[0] + correction_per_lap * np.arange(0, len(stint_laps))
    elif return_value == 'laptime':
        if reference_lap is None:
            # Ensure lapnumbers is not empty before accessing lapnumbers[0]
            if lapnumbers.size == 0:
                raise ValueError("Cannot determine reference lap from empty Laps object.")
            ref_lap_num: int = lapnumbers[0]
        else:
            ref_lap_num = reference_lap
        corrected_times = laptimes - correction_per_lap * (ref_lap_num - lapnumbers)
    else:
        raise ValueError("Unrecognised option for return_value. Use 'timedelta' or 'laptime'")

    return corrected_times


def fastest_drivers_in_team(session: Session) -> List[str]:
    """Find the fastest driver from each team in a session.

    Args:
        session (Session): FastF1 Session object containing session data.

    Returns:
        List[str]: List of driver codes for the fastest driver from each team.
    """
    if not session.laps_data_loaded:
        return [] # Or raise error
        
    teams: np.ndarray = np.unique(session.laps['Team'])
    fastest_drivers: List[str] = []
    for team in teams:
        fastest_lap: Optional[pd.Series] = session.laps.pick_team(team).pick_fastest()
        if fastest_lap is not None and 'Driver' in fastest_lap:
            fastest_drivers.append(fastest_lap['Driver'])
    return fastest_drivers


def get_compound_color(compound_name: str, session: Session) -> str:
    """Get the color code for a tyre compound.

    Args:
        compound_name (str): Name of the tyre compound.
        session (Session): FastF1 Session object for context (may influence colors).

    Returns:
        str: Hex color code for the tyre compound.
    """
    if compound_name == "TEST_UNKNOWN":
        color: str = "#434649" # Grey for unknown test tyres
    else:
        try:
            color = ff1plt.get_compound_color(compound_name, session)
        except Exception:
             # Fallback color if compound is unknown to fastf1
            color = "#808080" # Default grey
    return color


def finishing_order(session: Session) -> List[str]:
    """Determine the finishing order of drivers in a session.

    Handles different session types:
    - Race/Sprint: Based on final position and lap count
    - Qualifying: Based on Q1/Q2/Q3 results
    - Practice: Based on fastest lap times

    Args:
        session (Session): FastF1 Session object containing session data.

    Returns:
        List[str]: List of driver codes in finishing order.
    """
    session_name: str = session.session_info['Name']
    drivers_ordered: List[str] = []

    if session_name in ["Race", "Sprint"]:
        # Check necessary columns exist
        if not all(col in session.laps.columns for col in ["Driver", "Time", "LapNumber"]):
            return [] # Cannot determine order
            
        driver_finish_times: pd.DataFrame = session.laps[["Driver", "Time", "LapNumber"]].groupby("Driver", as_index=False).max()
        driver_finish_times = driver_finish_times.sort_values(by=["LapNumber", "Time"], ascending=[False, True])
        drivers_ordered = driver_finish_times["Driver"].tolist()

    elif session_name in ["Qualifying", "Sprint Qualifying"]:
        try:
            q1, q2, q3 = session.laps.pick_quicklaps().split_qualifying_sessions()
        except Exception:
             # Handle cases where split fails (e.g., not enough laps)
            return fastest_laps_in_session(session.laps)["Driver"].tolist() # Fallback to fastest lap
            
        q1_fastest: Laps = fastest_laps_in_session(q1)
        q2_fastest: Laps = fastest_laps_in_session(q2)
        q3_fastest: Laps = fastest_laps_in_session(q3)

        # Handle cases where Q sessions might be empty
        top: List[str] = q3_fastest["Driver"].tolist()
        middle: List[str] = q2_fastest[~q2_fastest["Driver"].isin(top)]["Driver"].tolist() # Drivers in Q2 but not Q3
        bottom: List[str] = q1_fastest[~q1_fastest["Driver"].isin(top + middle)]["Driver"].tolist() # Drivers in Q1 but not Q2/Q3
        
        drivers_ordered = top + middle + bottom

    else: # Practice session
        drivers_ordered = fastest_laps_in_session(session.laps).sort_values(by="LapTime", ascending=True)["Driver"].tolist()

    return drivers_ordered


def fix_positions(laps: Laps) -> Laps:
    """Fix position data for laps where FastF1 generated positions.

    Recalculates correct positions for laps where the original position data
    was generated by FastF1 rather than from official timing data.

    Args:
        laps (Laps): FastF1 Laps object containing lap data.

    Returns:
        Laps: FastF1 Laps object with corrected position data.
    """
    # Work on a copy
    laps_copy = laps.copy()
    
    # Check for required columns
    if not all(col in laps_copy.columns for col in ["FastF1Generated", "Position", "Time", "LapNumber"]):
        # Log warning or return original if columns missing
        return laps

    # Where cars stop: set 'Position' to NaN; set 'Time' to NaT 
    laps_copy.loc[laps_copy["FastF1Generated"], "Position"] = np.nan
    laps_copy.loc[laps_copy["FastF1Generated"], "Time"] = pd.NaT

    # Re-calculate the correct positions
    for lap_n in laps_copy['LapNumber'].unique():
        if pd.isna(lap_n):
            continue # Skip NaN lap numbers
            
        # get each drivers lap for the current lap number, sorted by time
        laps_eq_n = laps_copy.loc[
            laps_copy['LapNumber'] == lap_n, ('Time', 'Position')
        ].reset_index() # Keep index for sorting back
        
        if laps_eq_n.empty:
            continue
            
        laps_eq_n = laps_eq_n.sort_values(by='Time')

        # number positions and restore previous order by index
        laps_eq_n['Position'] = np.arange(1, len(laps_eq_n) + 1)
        
        # Use original index to update the main DataFrame slice
        laps_copy.loc[laps_eq_n['index'], 'Position'] = laps_eq_n['Position'].values

    return laps_copy


async def fix_grid_positions(session: Session) -> Session:
    """Fix missing grid positions for drivers in a session.

    For drivers without official grid positions, assigns positions based on
    qualifying results, placing them at the back of the grid.

    Args:
        session (Session): FastF1 Session object containing session data.

    Returns:
        Session: FastF1 Session object with corrected grid positions.
    """
    drivers: np.ndarray = pd.unique(session.laps['Driver'])
    # Ensure results are loaded for the session
    if not session.results_data_loaded:
        try:
            session.load_results() # Attempt to load results if missing
        except Exception:
             # Handle failure to load results
            return session
            
    try:
        # Assume f1_web returns DataFrame compatible with processing
        grid: pd.DataFrame = f1_web.process_starting_grid(f1_web.get_session_results(session, "starting-grid"))
    except Exception:
        # Handle failure to get/process grid
        return session

    # get drivers with no grid position in the processed grid data
    zero_grid_pos_drivers: np.ndarray = drivers[~np.isin(drivers, grid["Abbreviation"])]

    if zero_grid_pos_drivers.size == 0:
        return session # No drivers need fixing

    year: int = int(session.event.year)
    race: str = session.event['EventName']
    name: str = session.session_info['Name']

    quali_session_type: Optional[str] = None
    if name == "Race":
        quali_session_type = "Q"
    elif name == "Sprint Race":
        quali_session_type = "SQ" 
    
    if quali_session_type is None:
        return session # Cannot determine related quali session type

    try:
        quali_session: Session = await data.get_session(year, race, quali_session_type)
        if not quali_session.results_data_loaded:
            quali_session.load_results()
    except Exception:
        # Handle failure to load qualifying session
        return session
    
    # calculate their new position at the back of the grid
    quali_results: pd.DataFrame = quali_session.results

    select_list: List[bool] = [abbr in zero_grid_pos_drivers for abbr in quali_results["Abbreviation"]]
    # Ensure indices align if using boolean indexing directly
    missing_drivers_quali_results = quali_results.loc[select_list].copy()
    
    if missing_drivers_quali_results.empty:
        return session # No relevant drivers found in quali results

    last_nonzero_pos: int = len(drivers) - len(zero_grid_pos_drivers)
    missing_drivers_quali_results["NewGridPosition"] = np.arange(1, len(missing_drivers_quali_results) + 1) + last_nonzero_pos

    # write it to the session object
    # Ensure session.results is a DataFrame and has 'Abbreviation'
    if not isinstance(session.results, pd.DataFrame) or 'Abbreviation' not in session.results.columns:
        return session
        
    # Update using merge or map for safety, avoiding direct iteration
    update_map = missing_drivers_quali_results.set_index('Abbreviation')['NewGridPosition']
    session.results['GridPosition'] = session.results['GridPosition'].fillna(session.results['Abbreviation'].map(update_map))

    return session


def generate_plot_filename(
    plot_function_name: str,
    session: Session,  
    driver_codes: Optional[List[str]] = None,
    lap_numbers: Optional[List[int]] = None
) -> str:
    """Generates a standardized filename for plots.

    The filename concatenates the name of the plotting function (excluding the
    'plot_' prefix), the FastF1 event name, year, and session name (extracted
    from the session object). If driver codes or lap numbers are provided,
    they are also appended. All components are separated by underscores.
    The file extension '.png' is added.

    Args:
        plot_function_name (str): The name of the plotting function (e.g., 'plot_weather_data').
        session (Session): The FastF1 Session object from which event name,
            year, and session name are extracted.
        driver_codes (Optional[List[str]], optional): A list of driver abbreviations.
            Defaults to None.
        lap_numbers (Optional[List[int]], optional): A list of lap numbers.
            Defaults to None.

    Returns:
        str: The generated filename string (e.g.,
             'weather_data_Bahrain_Grand_Prix_2023_Race.png' or
             'telemetry_comparison_Italian_Grand_Prix_2024_Q_VER_LEC.png').
    """
    # Remove 'plot_' prefix if it exists
    name_part: str = plot_function_name.replace("plot_", "")

    # Extract info from session object
    event_name_str: str = session.event['EventName'].replace(" ", "_")
    year_str: str = str(session.event.year)
    session_name_str: str = session.session_info['Name'].replace(" ", "_")

    # Basic parts
    parts: List[str] = [
        name_part,
        event_name_str,
        year_str,
        session_name_str
    ]

    # Add driver codes if provided
    if driver_codes:
        parts.extend(list(driver_codes)) # Add driver codes

    # Add lap numbers if provided
    if lap_numbers:
        parts.extend([f"lap_{lap}" for lap in list(lap_numbers)]) # Add lap numbers

    return "_".join(parts) + ".png"
