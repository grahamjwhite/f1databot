import requests
import pandas as pd
import io

from fastf1.core import Session


def get_session_results(session: Session, data_type: str):

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

    raw_data["Abbreviation"] = raw_data.Driver.str.slice(start=-3)
    raw_data["cumu_time"] = raw_data.loc[:, ["Abbreviation", "Time"]].groupby('Abbreviation').cumsum()
    race_start = session.date 
    race_start_str = f'{race_start.year}/{race_start.month}/{race_start.day} '
    raw_data["time_of_day"] = pd.to_datetime(race_start_str + raw_data["Time of day"], format='%Y/%m/%d %H:%M:%S')
    raw_data["session_time"] = raw_data.time_of_day - race_start - session.session_info["GmtOffset"]
    raw_data = raw_data[raw_data.columns.drop(["No", "Driver", "Time of day"])]

    return raw_data

def process_starting_grid(raw_data: pd.DataFrame):
 
    raw_data["Abbreviation"] = raw_data.Driver.str.slice(start=-3)
    raw_data = raw_data[raw_data.columns.drop(["No", "Driver"])]

    return raw_data

def process_race_result(raw_data: pd.DataFrame):

    raw_data["Abbreviation"] = raw_data.Driver.str.slice(start=-3)
    raw_data = raw_data[raw_data.columns.drop(["No", "Driver"])]

    return raw_data

def process_qualifying_results(raw_data: pd.DataFrame):

    raw_data["Abbreviation"] = raw_data.Driver.str.slice(start=-3)
    raw_data = raw_data[raw_data.columns.drop(["No", "Driver"])]

    return raw_data