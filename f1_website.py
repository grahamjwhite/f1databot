"""Functions for retrieving and processing data from the official Formula 1 website.

This module provides tools to scrape and process various data types from the
official F1 results archive. It handles:

- Fetching raw data tables using requests and pandas
- Cleaning and formatting the retrieved data
- Processing specific data types like pit stops, starting grids, and results

The primary function `get_session_results` retrieves data based on session
information and the desired data type. Subsequent processing functions clean
and standardize the data for further analysis.

Note:
    Web scraping can be fragile and may break if the F1 website structure changes.

Dependencies:
    - requests: For making HTTP requests
    - pandas: For reading HTML tables and data manipulation
    - io: For handling string IO with pandas
    - fastf1: For Session object information
"""

import requests
import pandas as pd
import io

from fastf1.core import Session


def get_session_results(session: Session, data_type: str):
    """Fetch session results data from the Formula 1 website.

    Retrieves various types of session data from the official F1 website based on the
    specified data type. The function handles different data formats and cleans the
    resulting DataFrame.

    Args:
        session (Session): The F1 session to get results for.
        data_type (str): Type of data to retrieve. Options are:
            - 'pit-stop-summary'
            - 'race-result'
            - 'sprint-results'
            - 'fastest-laps'
            - 'starting-grid'
            - 'qualifying'
            - 'sprint-qualifying'
            - 'practice-3'
            - 'practice-2'
            - 'practice-1'

    Returns:
        pd.DataFrame: Cleaned DataFrame containing the requested session data.
            For race and sprint results, includes only rows with valid points.
    """

    # Possible data_types:
    # pit-stop-summary
    # race-result
    # sprint-results
    # fastest-laps
    # starting-grid
    # qualifying
    # sprint-qualifying
    # practice-3
    # practice-2
    # practice-1

    def countryMaker(country: str) -> str:
        split_words = country.lower().split()
        url_format = '-'.join(split_words)

        return url_format

    race_key = session.session_info["Meeting"]["Key"]
    country = session.session_info["Meeting"]["Country"]["Name"]
    year = session.event.year
    #baseURL = 'https://www.formula1.com/en/results/jcr:content/resultsarchive.html'
    baseURL = 'https://www.formula1.com/en/results'
    URLmaker = f'{baseURL}/{year}/races/{race_key}/{countryMaker(country)}/{data_type}'

    s = requests.Session()
    s.head(URLmaker)

    headers = {
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Accept-Language': 'en-GB,en;q=0.9,en-US;q=0.8,en-AU;q=0.7',
        'Cache-Control': 'no-cache',
        'Priority': 'u=1,i',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0',
        'X-Requested-With': 'XMLHttpRequest'
    }

    response = s.get(URLmaker, headers=headers)

    df = pd.read_html(io.StringIO(response.text), attrs={'class': 'f1-table'})
    df = pd.DataFrame(df[0])
    df = df[df.columns.drop(list(df.filter(regex='Unnamed')))] 

    # the website puts notes at the bottom of the table so this removes those rows
    if data_type in ["race-result", "sprint-results"]:
        df.loc[:, 'Pts'] = pd.to_numeric(df['Pts'], errors='coerce') 
        df = df.loc[df["Pts"].notnull(), :]

    return df

def process_pit_stop_summary(raw_data: pd.DataFrame, session: Session):
    """Process and clean pit stop summary data.

    Processes raw pit stop data by:
    - Extracting driver abbreviations
    - Calculating cumulative pit stop times
    - Converting time of day to session time
    - Removing unnecessary columns

    Args:
        raw_data (pd.DataFrame): Raw pit stop data from get_session_results.
        session (Session): The F1 session the data is from.

    Returns:
        pd.DataFrame: Processed pit stop data with:
            - Driver abbreviations
            - Cumulative pit stop times
            - Session time for each stop
            - Cleaned column structure
    """

    raw_data["Abbreviation"] = raw_data.Driver.str.slice(start=-3)
    raw_data["cumu_time"] = raw_data.loc[:, ["Abbreviation", "Time"]].groupby('Abbreviation').cumsum()
    race_start = session.date 
    race_start_str = f'{race_start.year}/{race_start.month}/{race_start.day} '
    raw_data["time_of_day"] = pd.to_datetime(race_start_str + raw_data["Time of day"], format='%Y/%m/%d %H:%M:%S')
    raw_data["session_time"] = raw_data.time_of_day - race_start - session.session_info["GmtOffset"]
    raw_data = raw_data[raw_data.columns.drop(["No", "Driver", "Time of day"])]

    return raw_data

def process_starting_grid(raw_data: pd.DataFrame):
    """Process and clean starting grid data.

    Processes raw starting grid data by:
    - Extracting driver abbreviations
    - Removing unnecessary columns (driver numbers and full names)

    Args:
        raw_data (pd.DataFrame): Raw starting grid data from get_session_results.

    Returns:
        pd.DataFrame: Processed starting grid data with:
            - Driver abbreviations
            - Cleaned column structure
    """

    raw_data["Abbreviation"] = raw_data.Driver.str.slice(start=-3)
    raw_data = raw_data[raw_data.columns.drop(["No", "Driver"])]

    return raw_data

def process_race_result(raw_data: pd.DataFrame):
    """Process and clean race result data.

    Processes raw race result data by:
    - Extracting driver abbreviations
    - Removing unnecessary columns (driver numbers and full names)

    Args:
        raw_data (pd.DataFrame): Raw race result data from get_session_results.

    Returns:
        pd.DataFrame: Processed race result data with:
            - Driver abbreviations
            - Cleaned column structure
    """

    raw_data["Abbreviation"] = raw_data.Driver.str.slice(start=-3)
    raw_data = raw_data[raw_data.columns.drop(["No", "Driver"])]

    return raw_data

def process_qualifying_results(raw_data: pd.DataFrame):
    """Process and clean qualifying results data.

    Processes raw qualifying results data by:
    - Extracting driver abbreviations
    - Removing unnecessary columns (driver numbers and full names)

    Args:
        raw_data (pd.DataFrame): Raw qualifying results data from get_session_results.

    Returns:
        pd.DataFrame: Processed qualifying results data with:
            - Driver abbreviations
            - Cleaned column structure
    """

    raw_data["Abbreviation"] = raw_data.Driver.str.slice(start=-3)
    raw_data = raw_data[raw_data.columns.drop(["No", "Driver"])]

    return raw_data