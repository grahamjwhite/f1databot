import fastf1
import fastf1.plotting
import fastf1.core
import numpy as np
import pandas as pd
import util_functions as uf
import f1_website as f1web
import timple.timedelta as tmpldelta
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm

from fastf1.core import Laps
from timple.timedelta import strftimedelta
from timple.timedelta import TimedeltaFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from classes import ValidationException
from patsy import dmatrices
from constants import F1DATABOT_PLOT_CACHE


def get_custom_driver_styles(chart_type: str = 'line') -> dict: 

    style_dict = {
        'line': [
            {'linestyle': 'solid', 'color': 'auto'},
            {'linestyle': '--', 'color': 'auto'},
            {'linestyle': '-', 'color': 'auto'}
        ],
        'line_marker': [
            {'linestyle': 'solid', 'color': 'auto', 'marker': 'o'},
            {'linestyle': 'solid', 'color': 'auto', 'marker': '^'},
            {'linestyle': 'solid', 'color': 'auto', 'marker': 'D'}
        ],
        'bar': [
            {'color': 'auto'},
            {'color': 'auto', 'hatch': '//'},
            {'color': 'auto', 'hatch': '/'}  
        ],
        'scatter': [
            {'color': 'auto', 'marker': 'o'},
            {'color': 'auto', 'marker': '^'},
            {'color': 'auto', 'marker': 'D'}
        ]
    }

    return style_dict[chart_type]


def check_plot_params(params: dict) -> dict:

    for param in params.keys():
        if param == 'driver1' or param == 'driver2':
            params[param] = params[param].upper()

    return(params)


def plot_weather(session: fastf1.core.Session, save: bool=False) -> None:

    fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')

    weather = session.weather_data
    track_degrees = weather['TrackTemp'].to_numpy()
    ambient_degrees = weather['AirTemp'].to_numpy()
    rain = session.weather_data["Rainfall"].to_numpy()

    time = weather['Time'].dt.total_seconds()/60

    fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[4,1], figsize=(10,7), sharex=True, gridspec_kw={'hspace': 0.05})
    ax1.plot(time, track_degrees, color = 'dodgerblue', label = 'Track temperature')
    ax1.plot(time, ambient_degrees, color = 'gold', label = 'Air temperature')
    ax1.set_ylabel('Temperature (degrees celcius)')
    ax1.legend()

    ax2.plot(time, rain, color='blue')
    ax2.set_xlabel('Session time (minutes)')
    ax2.set(yticklabels=[])   
    ax2.set_ylabel("Rain")
    ax2.tick_params(left=False)
    ax2.set_yticks((0,1), ["No", "Yes"])

    plt.suptitle("Session Weather \n"
                 f"{session.event['EventName']} {session.event.year} {session.session_info['Name']}",
                 fontsize='xx-large')

    if save==True:
        plt.savefig(f"{F1DATABOT_PLOT_CACHE}/weather {session.event['EventName']} {session.event.year} {session.session_info['Name']}.png", 
                    format='png', bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def plot_track_layout(session: fastf1.core.Session, save:bool = False) -> None:

    fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')

    lap = session.laps.pick_fastest()
    pos = lap.get_pos_data()

    circuit_info = session.get_circuit_info()

    # Get an array of shape [n, 2] where n is the number of points and the second
    # axis is x and y.
    track = pos.loc[:, ('X', 'Y')].to_numpy()/10

    # Convert the rotation angle from degrees to radian.
    track_angle = circuit_info.rotation / 180 * np.pi

    # Rotate and plot the track map.
    rotated_track = uf.rotate(track, angle=track_angle)
    plt.plot(rotated_track[:, 0], rotated_track[:, 1], color='DodgerBlue')

    offset_vector = [50, 0]  # offset length is chosen arbitrarily to 'look good'

    # Iterate over all corners.
    for _, corner in circuit_info.corners.iterrows():
        # Create a string from corner number and letter
        txt = f"{corner['Number']}{corner['Letter']}"

        # Convert the angle from degrees to radian.
        offset_angle = corner['Angle'] / 180 * np.pi

        # Rotate the offset vector so that it points sideways from the track.
        offset_x, offset_y = uf.rotate(offset_vector, angle=offset_angle)

        # Add the offset to the position of the corner
        text_x = corner['X']/10 + offset_x
        text_y = corner['Y']/10 + offset_y

        # Rotate the text position equivalently to the rest of the track map
        text_x, text_y = uf.rotate([text_x, text_y], angle=track_angle)

        # Rotate the center of the corner equivalently to the rest of the track map
        track_x, track_y = uf.rotate([corner['X']/10, corner['Y']/10], angle=track_angle)

        # Draw a circle next to the track.
        plt.scatter(text_x, text_y, color='grey', s=140)

        # Draw a line from the track to this circle.
        plt.plot([track_x, text_x], [track_y, text_y], color='grey')

        # Finally, print the corner number inside the circle.
        plt.text(text_x, text_y, txt,
                va='center_baseline', ha='center', size='small', color='white')
        
    plt.title(session.event['Location'])
    plt.axis('off')
    
    if save==True:
        plt.savefig(f"{F1DATABOT_PLOT_CACHE}/track_map_{session.event['EventName']}_{session.event.year}.png", 
                    format='png', bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def plot_fastest_laps(session: fastf1.core.Session, save:bool = False) -> None:

    fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')

    laps = session.laps.pick_quicklaps(threshold=1.15)
    fastest_laps = uf.fastest_laps_in_session(laps)

    # plot the data
    fig, ax = plt.subplots(figsize=(10, 8))

    for index, lap in fastest_laps.iterlaps():
        style = fastf1.plotting.get_driver_style(identifier = lap['Driver'],
                                                 style = get_custom_driver_styles('bar'),
                                                 session=session)
        ax.barh(lap['Driver'], lap['LapTimeDelta'], edgecolor = 'black',
                **style)

    ax.set_yticks(fastest_laps.index, minor=False)
    ax.set_yticklabels(fastest_laps['Driver'])

    # show fastest at the top
    ax.invert_yaxis()

    # draw vertical lines behind the bars
    ax.set_axisbelow(True)
    ax.grid(linewidth=0.5, alpha=0.5, axis='x')
    ax.set_xlabel("Per cent difference to fastest session time")

    pole_lap = fastest_laps.pick_fastest()

    # draw labels for lap times/deltas
    for index, lap in fastest_laps.iterlaps():
        if lap['LapTime'] == pole_lap['LapTime']:
            ax.text(0, 0.2, strftimedelta(lap['LapTime'], '%m:%s.%ms'))
        else:
            ax.text(lap['LapTimeDelta'], index+0.2, f"+{round(lap['LapTime'].total_seconds() - pole_lap['LapTime'].total_seconds(), 3)}s")

    plt.suptitle("Fastest Lap Times \n"
                 f"{session.event['EventName']} {session.event.year} {session.session_info['Name']}", 
                 fontsize = 'xx-large')

    if save==True:
        plt.savefig(f"{F1DATABOT_PLOT_CACHE}/fastest_laps {session.event['EventName']} {session.event.year} {session.session_info['Name']}.png", 
                    format='png', bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def plot_qualifying_fastest_laps(session: fastf1.core.Session, save:bool = False) -> None:

    q1, q2, q3 = session.laps.pick_quicklaps(threshold=1.15).split_qualifying_sessions()
    q1 = uf.fastest_laps_in_session(q1)
    q2 = uf.fastest_laps_in_session(q2)
    q3 = uf.fastest_laps_in_session(q3)

    fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme = 'fastf1')

    # plot the data
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(12, 36, figure = fig, hspace = 0)

    ax1 = plt.subplot(gs.new_subplotspec((1, 0), colspan = 11, rowspan = 11))
    ax2 = fig.add_subplot(gs[1:,12:23])
    ax3 = fig.add_subplot(gs[1:,24:36])

    def plot_q_session(qdata: fastf1.core.Laps, ax: plt.axis, title: str) -> None:

        for idx, lap in qdata.iterlaps():
            style = fastf1.plotting.get_driver_style(lap['Driver'],
                                                     get_custom_driver_styles('bar'),
                                                     session)
            ax.barh(lap['Driver'], lap['LapTimeDelta'],
                    **style, edgecolor='black')
        
        ax.set_yticks(qdata.index)
        ax.set_yticklabels(qdata['Driver'])
        ax.invert_yaxis()
        ax.set_axisbelow(True)
        ax.set_xlabel("Per cent difference to fastest")
        pole_lap = qdata.pick_fastest()
        ax.set_title(title, loc='center', size=12)
        ax.set_xlim(xmax=max(qdata['LapTimeDelta'])+0.25)
        ax.grid(linewidth=0.5, alpha=0.5, axis='x')

        # draw labels for lap times/deltas
        for index, lap in qdata.iterlaps():
            if lap['LapTime'] == pole_lap['LapTime']:
                ax.text(0.02, 0.2, strftimedelta(lap['LapTime'], '%m:%s.%ms'))
            else:
                ax.text(lap['LapTimeDelta']+0.02, index+0.2, f"+{round(lap['LapTime'].total_seconds() - pole_lap['LapTime'].total_seconds(), 3)}s")

    plot_q_session(q1, ax1, 'Q1')
    plot_q_session(q2, ax2, 'Q2')
    plot_q_session(q3, ax3, 'Q3')

    plt.suptitle(f"Qualifying Session Times \n"
                 f"{session.event['EventName']} {session.event.year} {session.session_info['Name']}",
                 fontsize='xx-large')
    
    if save==True:
        plt.savefig(f"{F1DATABOT_PLOT_CACHE}/qualifying_fastest_laps {session.event['EventName']} {session.event.year} {session.session_info['Name']}.png", 
                    format='png', bbox_inches='tight')
        plt.clf()
    else:
        plt.show()

    

def plot_tyre_strategies(session: fastf1.core.Session, save:bool = False) -> None:

    fastf1.plotting.setup_mpl(misc_mpl_mods = False, color_scheme = 'fastf1')

    drivers = np.unique(session.laps['Driver'])
    stints = session.laps[["Driver", "Stint", "Compound", "LapNumber", "LapTime", "LapStartTime", "Time", "FreshTyre", "PitOutTime", "PitInTime"]]

    stint_dict = dict()
    for driver in drivers:
        stint_dict.update({driver: np.unique(stints.loc[stints['Driver']==driver, 'Stint'])})

    stint_plot_data = dict()
    for driver in drivers:
        for stint in stint_dict[driver]:
            stint_laps = stints.loc[(stints['Driver']==driver) & (stints['Stint']==stint), :]

            if len(stint_laps) > 0:

                start_stint_lapnum = np.min(stint_laps['LapNumber'])
                end_stint_lapnum = np.max(stint_laps['LapNumber'])
                stint_num_laps = end_stint_lapnum - start_stint_lapnum + 1
                start_stint_lap = stint_laps.loc[stint_laps['LapNumber']==start_stint_lapnum, :]
                end_stint_lap = stint_laps.loc[stint_laps['LapNumber']==end_stint_lapnum, :]
                start_stint_time = start_stint_lap["LapStartTime"] if np.isnan(start_stint_lap["PitOutTime"].values[0]) else start_stint_lap["PitOutTime"]
                start_stint_time = start_stint_time.dt.total_seconds().values[0]/60
                end_stint_time = end_stint_lap["Time"] if np.isnan(end_stint_lap["PitInTime"].values[0]) else end_stint_lap["PitInTime"]
                end_stint_time = end_stint_time.dt.total_seconds().values[0]/60
                stint_time = end_stint_time - start_stint_time
                freshtyre = stint_laps.FreshTyre.iat[0]
                compound = stint_laps.Compound.iat[0]
                hatching = "" if freshtyre == True else "///"

                stint_plot_data.update({f"{driver}_{int(stint)}": {'driver': driver,
                                                            'stint': stint,
                                                            'data': {
                                                                'start_stint_lap': start_stint_lapnum,
                                                                'end_stint_lap': end_stint_lapnum,
                                                                'stint_num_laps': stint_num_laps,
                                                                'start_stint_time': start_stint_time,
                                                                'end_stint_time': end_stint_time,
                                                                'stint_time': stint_time,
                                                                'freshtyre': freshtyre,
                                                                'compound': compound,
                                                                'hatching': hatching}}
                                        })

    # get the finishing order
    drivers_ordered = uf.finishing_order(session)

    fig, ax = plt.subplots(figsize=(10, 10))

    for driver in drivers_ordered:

        driver_stints = {key: value for (key, value) in stint_plot_data.items() if driver in key}

        for key, stint in driver_stints.items():

            # each row contains the compound name and stint length
            # we can use these information to draw horizontal bars

            if session.session_info["Name"] in ["Race", "Sprint Race"]:
                plt.barh(
                    y=stint['driver'],
                    width=stint['data']['stint_num_laps']+1,
                    left=stint['data']['start_stint_lap']-1,
                    color=uf.get_compound_color(stint['data']['compound'], session),
                    edgecolor="black",
                    fill=True,
                    hatch=stint['data']['hatching']
                )

                y_text = stint['driver']
                x_text = stint['data']['start_stint_lap']-1 + (stint['data']['stint_num_laps'])/2
                plt.scatter(y=y_text, x=x_text, color='lightgrey', s=200)
                plt.text(y=y_text, x=x_text, s = int(stint['data']['stint_num_laps']),
                    va='center_baseline', ha='center', size='small', color='black')

                plt.xlabel("Lap number")

            else: 
                plt.barh(
                    y=stint['driver'],
                    width=stint['data']['stint_time'],
                    left=stint['data']['start_stint_time'],
                    color=uf.get_compound_color(stint['data']['compound'], session),
                    edgecolor="black",
                    fill=True,
                    hatch=stint['data']['hatching']
                )

                y_text = stint['driver']
                x_text = stint['data']['start_stint_time'] + (stint['data']['stint_time'])/2
                plt.scatter(y=y_text, x=x_text, color='lightgrey', s=200)
                plt.text(y=y_text, x=x_text, s = int(stint['data']['stint_num_laps']),
                    va='center_baseline', ha='center', size='small', color='black')

                plt.xlabel("Session time (minutes)")

    plt.title("Tyre Strategy \n"
            f"{session.event['EventName']} {session.event.year} {session.session_info['Name']}",
            fontsize='xx-large')

    labels = stints['Compound'].unique()

    handles = []
    for label in labels:
        handles.append(mpatches.Patch(color = uf.get_compound_color(label, session), label = label))

    handles.append(mpatches.Patch(facecolor="White", fill=True, hatch="///", label="Used tyre"))

    plt.legend(handles = handles, ncols = len(labels)+1, loc = "lower left", bbox_to_anchor = (0,0.95))
    
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if save==True:
        plt.savefig(f"{F1DATABOT_PLOT_CACHE}/tyre_strategy_{session.event['EventName']}_{session.event.year}_{session.session_info['Name']}.png", 
                    format='png', bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def plot_telemetry_comparison(session: fastf1.core.Session, driver1: str, driver2: str, 
                              lap:(int|None) = None, save:bool = False) -> None:

    fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')

    # get circuit info for the corner markers
    circuit_info = session.get_circuit_info()

    # get the two laps
    if lap == None:
        driver1_lap = session.laps.pick_drivers(driver1).pick_fastest()
        driver2_lap = session.laps.pick_drivers(driver2).pick_fastest()
    else:
        driver1_lap = session.laps.pick_drivers(driver1).pick_laps(lap)
        driver2_lap = session.laps.pick_drivers(driver2).pick_laps(lap)    

    delta_time, ref_tel, compare_tel = fastf1.utils.delta_time(driver1_lap, driver2_lap)

    # get the telemetry and add distance
    driver1_tel = driver1_lap.get_car_data(interpolate_edges=True).add_distance()
    driver2_tel = driver2_lap.get_car_data(interpolate_edges=True).add_distance()

    # calculate a DRS indicator
    driver1_tel['DRS_indicator'] = driver1_tel['DRS'].apply(lambda x: 1 if (x in [10, 12, 14]) else 0)
    driver2_tel['DRS_indicator'] = driver2_tel['DRS'].apply(lambda x: 1 if (x in [10, 12, 14]) else 0)

    # plot the traces
    driver1_style = fastf1.plotting.get_driver_style(driver1,
                                                     get_custom_driver_styles('line'),
                                                     session)
    driver2_style= fastf1.plotting.get_driver_style(driver2,
                                                     get_custom_driver_styles('line'),
                                                     session)

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, height_ratios=[2,1,1,1,1,1], figsize=(15,18),
                                                       sharex=True)
    
    plt.subplots_adjust(hspace=0.05)

    ax1.plot(driver1_tel['Distance'], driver1_tel['Speed'], **driver1_style, label = driver1)
    ax1.plot(driver2_tel['Distance'], driver2_tel['Speed'], **driver2_style, label = driver2, alpha=0.8)
    ax1.grid(linestyle='-', linewidth=0.2, axis = 'y')
    ymin, ymax = ax1.get_ylim()
    ax1.vlines(x = circuit_info.corners['Distance'], ymin = 0, ymax = 1, transform = ax1.get_xaxis_transform(),
            linestyle = '--', colors = 'grey', linewidth = 1)

    ax2.plot(ref_tel['Distance'], delta_time, color = "White", label = driver1)
    ax2.grid(linestyle='-', linewidth=0.2, axis = 'y')
    ax2.vlines(x=circuit_info.corners['Distance'], ymin = delta_time.min(), ymax = 1, transform = ax2.get_xaxis_transform(),
            linestyle = '--', colors = 'grey', linewidth = 1)

    ax3.plot(driver1_tel['Distance'], driver1_tel['Throttle'], **driver1_style, label = driver1)
    ax3.plot(driver2_tel['Distance'], driver2_tel['Throttle'], **driver2_style, label = driver2, alpha=0.8)
    ax3.grid(linestyle='-', linewidth=0.2, axis = 'y')
    ax3.vlines(x=circuit_info.corners['Distance'], ymin = 0, ymax = 1, transform = ax3.get_xaxis_transform(),
            linestyle = '--', colors = 'grey', linewidth = 1)

    ax4.plot(driver1_tel['Distance'], driver1_tel['Brake'], **driver1_style, label = driver1)
    ax4.plot(driver2_tel['Distance'], driver2_tel['Brake'], **driver2_style, label = driver2, alpha=0.8)
    ax4.grid(linestyle='-', linewidth=0.2, axis = 'y')
    ax4.vlines(x=circuit_info.corners['Distance'], ymin = 0, ymax = 1, transform = ax4.get_xaxis_transform(),
            linestyle = '--', colors = 'grey', linewidth = 1)

    ax5.plot(driver1_tel['Distance'], driver1_tel['DRS_indicator'], **driver1_style, label = driver1)
    ax5.plot(driver2_tel['Distance'], driver2_tel['DRS_indicator'], **driver2_style, label = driver2, alpha=0.8)
    ax5.grid(linestyle='-', linewidth=0.2, axis = 'y')
    ax5.vlines(x=circuit_info.corners['Distance'], ymin = 0, ymax = 1, transform = ax5.get_xaxis_transform(),
            linestyle = '--', colors = 'grey', linewidth = 1)
    
    ax6.plot(driver1_tel['Distance'], driver1_tel['nGear'], **driver1_style, label = driver1)
    ax6.plot(driver2_tel['Distance'], driver2_tel['nGear'], **driver2_style, label = driver2, alpha=0.8)
    ax6.grid(linestyle='-', linewidth=0.2, axis = 'y')
    ax6.vlines(x=circuit_info.corners['Distance'], ymin = 0, ymax = 1, transform = ax6.get_xaxis_transform(),
            linestyle = '--', colors = 'grey', linewidth = 1)

    ax1.set_ylabel('Speed (km/h)')
    ax2.set_ylabel('Delta (s)')
    ax3.set_ylabel('Throttle (%)')
    ax4.set_ylabel('Brake (On/Off)')
    ax4.set_yticks((0, 1), ["Off", "On"])
    ax5.set_ylabel('DRS (On/Off)')
    ax5.set_yticks((0, 1), ["Off", "On"])
    ax6.set_xlabel('Distance (m)')
    ax6.set_ylabel('Gear')

    for i, ax in enumerate(fig.axes):
        ax.tick_params(axis='x', which='both', length=0)

    ax1.legend(loc = 'lower left', fontsize='large')

    if lap is None:
        driver1_lapstring = "No laptime recorded" if pd.isnull(driver1_lap['LapTime']) else strftimedelta(driver1_lap['LapTime'], '%m:%s.%ms')
        driver2_lapstring = "No laptime recorded" if pd.isnull(driver2_lap['LapTime']) else strftimedelta(driver2_lap['LapTime'], '%m:%s.%ms')
    else:
        driver1_lapstring = "No laptime recorded" if pd.isnull(driver1_lap['LapTime'].item()) else strftimedelta(driver1_lap['LapTime'].item(), '%m:%s.%ms')
        driver2_lapstring = "No laptime recorded" if pd.isnull(driver2_lap['LapTime'].item()) else strftimedelta(driver2_lap['LapTime'].item(), '%m:%s.%ms')

    if lap is None:
        lap_text = "Fastest Lap"
    else:
        lap_text = f"Lap {lap}"

    plt.suptitle(f"Telemetry Comparison \n "
                f"{session.event['EventName']} {session.event.year} {session.session_info['Name']} {lap_text} \n"
                f"{driver1}: {driver1_lapstring} \n" 
                f"{driver2}: {driver2_lapstring} \n",
                fontsize = 'xx-large', y=0.95)

    # Plot the corner number just above each vertical line.
    # For corners that are very close together, the text may overlap. A more
    # complicated approach would be necessary to reliably prevent this.
    v_max = np.max([driver1_tel['Speed'].max(), driver2_tel['Speed'].max()])
    ax1.text(0, ymax, "Corner:", va='bottom', ha='right', size='small')
    for _, corner in circuit_info.corners.iterrows():
        txt = f"{corner['Number']}{corner['Letter']}"
        ax1.text(corner['Distance'], ymax, txt,
                va='bottom', ha='center', size='small')
        
    if save==True:
        plt.savefig(f"{F1DATABOT_PLOT_CACHE}/compare_telemetry {session.event['EventName']} {session.event.year} {session.session_info['Name']} {driver1} {driver2} {lap_text}.png", 
                    format='png', bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def plot_corner_analysis(session: fastf1.core.Session, driver1: str, driver2: str, corner:int, lower_offset:int = 200,
                         upper_offset:int = 200, lap_number:(int|None) = None, save: bool = False) -> None:
    
    fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme = 'fastf1')

    # Get the laps and circuit info
    laps = session.laps
    circuit_info = session.get_circuit_info()

    distance_to_corner = circuit_info.corners['Distance'][circuit_info.corners['Number'] == corner].item()
    distance_min = distance_to_corner - lower_offset
    distance_max = distance_to_corner + upper_offset

    # Extracting the laps

    if lap_number is None:
        laps_driver1 = laps.pick_drivers(driver1).pick_fastest()
        laps_driver2 = laps.pick_drivers(driver2).pick_fastest()
        telemetry_driver1 = laps_driver1.get_car_data().add_distance()
        telemetry_driver2 = laps_driver2.get_car_data().add_distance()
    else:
        laps_driver1 = laps.pick_drivers(driver1).pick_laps(lap_number)
        laps_driver2 = laps.pick_drivers(driver2).pick_laps(lap_number)  
        telemetry_driver1 = laps_driver1.get_car_data().add_distance()
        telemetry_driver2 = laps_driver2.get_car_data().add_distance()


    # Assigning labels to what the drivers are currently doing 
    telemetry_driver1.loc[(telemetry_driver1['Brake'] > 0), 'CurrentAction'] = 'Brake'
    telemetry_driver1.loc[(telemetry_driver1['Brake'] > 0) & (telemetry_driver1['Throttle'] > 0), 'CurrentAction'] = 'Brake and throttle'
    telemetry_driver1.loc[(telemetry_driver1['Throttle'] > 95) & (telemetry_driver1['Brake'] == 0), 'CurrentAction'] = 'Full Throttle'
    telemetry_driver1.loc[(telemetry_driver1['Brake'] == 0) & (telemetry_driver1['Throttle'] <= 95), 'CurrentAction'] = 'Part throttle'

    telemetry_driver2.loc[(telemetry_driver2['Brake'] > 0), 'CurrentAction'] = 'Brake'
    telemetry_driver2.loc[(telemetry_driver2['Brake'] > 0) & (telemetry_driver2['Throttle'] > 0), 'CurrentAction'] = 'Brake and throttle'
    telemetry_driver2.loc[(telemetry_driver2['Throttle'] == 100) & (telemetry_driver2['Brake'] == 0), 'CurrentAction'] = 'Full Throttle'
    telemetry_driver2.loc[(telemetry_driver2['Brake'] == 0) & (telemetry_driver2['Throttle'] < 100), 'CurrentAction'] = 'Part throttle'

    # Numbering each unique action to identify changes, so that we can group later on
    telemetry_driver1['ActionID'] = (telemetry_driver1['CurrentAction'] != telemetry_driver1['CurrentAction'].shift(1)).cumsum()
    telemetry_driver2['ActionID'] = (telemetry_driver2['CurrentAction'] != telemetry_driver2['CurrentAction'].shift(1)).cumsum()

    # Identifying all unique actions
    actions_driver1 = telemetry_driver1[['ActionID', 'CurrentAction', 'Distance']].groupby(['ActionID', 'CurrentAction']).max('Distance').reset_index()
    actions_driver2 = telemetry_driver2[['ActionID', 'CurrentAction', 'Distance']].groupby(['ActionID', 'CurrentAction']).max('Distance').reset_index()

    actions_driver1['Driver'] = driver1
    actions_driver2['Driver'] = driver2

    # Calculating the distance between each action, so that we know how long the bar should be
    actions_driver1['DistanceDelta'] = actions_driver1['Distance'] - actions_driver1['Distance'].shift(1)
    actions_driver1.loc[0, 'DistanceDelta'] = actions_driver1.loc[0, 'Distance']

    actions_driver2['DistanceDelta'] = actions_driver2['Distance'] - actions_driver2['Distance'].shift(1)
    actions_driver2.loc[0, 'DistanceDelta'] = actions_driver2.loc[0, 'Distance']

    # Merging together
    all_actions = pd.concat([actions_driver1, actions_driver2])

    # get the time delta
    delta_time, ref_tel, compare_tel = fastf1.utils.delta_time(laps_driver1, laps_driver2)

    distance = ref_tel['Distance'][(ref_tel['Distance'] >= distance_min) & (ref_tel['Distance'] <= distance_max)]
    delta = delta_time[(ref_tel['Distance'] >= distance_min) & (ref_tel['Distance'] <= distance_max)].reset_index(drop='index')

    delta_gain = delta[len(delta)-1] - delta[0]

    if delta_gain > 0:
        delta_text = f"{driver1} gained {round(delta_gain, 3)} seconds"
    else:
        delta_text = f"{driver2} gained {round(abs(delta_gain), 3)} seconds"

    # Do the plot

    telemetry_colors = {
        'Brake': 'red',
        'Brake and throttle': 'orange',
        'Part throttle': 'grey',
        'Full Throttle': 'green'
    }

    fig = plt.figure(figsize=(13, 11))
    gs = GridSpec(11,13, figure = fig, hspace = 0) # 

    ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan = 13, rowspan = 6))
    ax2 = fig.add_subplot(gs[7:8,:], sharex = ax1)
    ax3 = fig.add_subplot(gs[9:11,:], sharex = ax1)

    # Lineplot for speed 
    driver1_style = fastf1.plotting.get_driver_style(driver1, get_custom_driver_styles('line'), session)
    driver2_style = fastf1.plotting.get_driver_style(driver2, get_custom_driver_styles('line'), session)
    ax1.plot(telemetry_driver1['Distance'], telemetry_driver1['Speed'], label=driver1, **driver1_style)
    ax1.plot(telemetry_driver2['Distance'], telemetry_driver2['Speed'], label=driver2, **driver2_style)

    ax1.set(ylabel='Speed')
    ax1.legend(loc="lower right")

    # Horizontal barplot for driver actions
    for driver in [driver1, driver2]:
        driver_actions = all_actions.loc[all_actions['Driver'] == driver]
        
        previous_action_end = 0
        for _, action in driver_actions.iterrows():
            ax2.barh(
                [driver], 
                action['DistanceDelta'], 
                left=previous_action_end, 
                color=telemetry_colors[action['CurrentAction']]
            )
            
            previous_action_end = previous_action_end + action['DistanceDelta']


    # Line plot of lap delta
    ax3.plot(distance, delta, color='white')
    ax3.set_title("Laptime delta")
    ax3.grid(linewidth=0.5, linestyle='--')
    ax3.set_ylabel("delta (s)")
            
    # Styling of the plot
    plt.xlabel('Distance')

    # Remove frame from plot
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # Add legend
    labels = list(telemetry_colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=telemetry_colors[label]) for label in labels]
    ax2.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.2, 1.05), ncol=4)

    # Zoom in on the specific part we want to see
    ax1.set_xlim(distance_min, distance_max)
    ax2.set_xlim(distance_min, distance_max)

    # add gridlines
    ax1.grid(linewidth=0.5, linestyle='--')

    # generate text for the lap
    if lap_number == None:
        lap_text = 'Fastest Lap'
    else:
        lap_text = f'Lap Number {lap_number}'

    plt.suptitle(f"Cornering Analysis - {session.event['EventName']} {session.event.year} {session.session_info['Name']} \n"
                f"{lap_text} - Corner {corner}\n"
                f"{driver1} versus {driver2} - {delta_text}",
                fontsize = 'xx-large')

    if save==True:
        plt.savefig(f"{F1DATABOT_PLOT_CACHE}/Corner analysis {session.event['EventName']} {session.event.year} {session.session_info['Name']} {driver1} {driver2} {lap_text} Corner {corner}.png", 
                    format='png', bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def plot_gears_on_track(session: fastf1.core.Session, save: bool = False) -> None:

    lap = session.laps.pick_fastest()
    tel = lap.get_car_data()
    pos = lap.get_pos_data()

    x = np.array(pos['X'].values)
    y = np.array(pos['Y'].values)

    points = np.array([x, y]).T.reshape(-1, 1, 2)

    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    gear = tel['nGear'].to_numpy().astype(float)

    cmap = mpl.colormaps['Paired']
    lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N+1), cmap=cmap)
    lc_comp.set_array(gear)
    lc_comp.set_linewidth(4)

    fig, ax = plt.subplots(figsize=(10,10))

    ax.add_collection(lc_comp)
    ax.set_xlim(x.min()-100, x.max()+100)
    ax.set_ylim(y.min()-100, y.max()+100)

    ax.set_title(f"Gear Shift Visualization\n"
                f"{session.event['EventName']} {session.event.year} - {lap['Driver']}")

    cbar = fig.colorbar(mappable=lc_comp, label="Gear", boundaries=np.arange(1, 10))
    cbar.set_ticks(np.arange(1.5, 9.5))
    cbar.set_ticklabels(np.arange(1, 9))

    if save==True:
        plt.savefig(f"{F1DATABOT_PLOT_CACHE}/Gears on track {session.event['EventName']} {session.event.year}.png", 
                    format='png', bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def plot_speed_versus_laptime(session: fastf1.core.Session, trap_loc:(str|None), save: bool = False) -> None:

    fastf1.plotting.setup_mpl(mpl_timedelta_support=True, misc_mpl_mods=True, color_scheme='fastf1')

    laps = session.laps.pick_quicklaps(threshold=1.15)

    # get a list of all drivers
    drivers = pd.unique(laps['Driver'])

    # get the fastest lap for each driver and their top speed
    list_fastest_laps = list()
    for drv in drivers:
        drvs_fastest_lap = laps.pick_drivers(drv).pick_fastest()
        list_fastest_laps.append(drvs_fastest_lap)

    fastest_laps = Laps(list_fastest_laps).sort_values(by='LapTime').reset_index(drop=True)

    # remove any NA values
    fastest_laps = fastest_laps.dropna(subset="Driver")

    # get the fastest speed for each driver
    top_speed = []
    for driver in fastest_laps['Driver']:
        tel = laps.pick_drivers(driver).pick_fastest().get_car_data()
        top_speed.append(max(tel['Speed']))
    fastest_laps['Top_speed'] = top_speed

    # remove any drivers with top speeds more than 10% below the median
    avg_speed = np.median(top_speed)
    mask = fastest_laps['Top_speed'] < 0.9*avg_speed
    fastest_laps = fastest_laps.loc[~mask].reset_index()

    # create handles for the legend
    drivers = pd.unique(fastest_laps['Driver'])
    handles = []
    for driver in drivers:
        handles.append(mpatches.Patch(color = fastf1.plotting.get_driver_color(driver, session), label = driver))

    if trap_loc == 'FL':
        speed = fastest_laps['SpeedFL']
        xlab = "Speed trap at finish line"
    elif trap_loc == 'ST':
        speed = fastest_laps['SpeedST']
        xlab = "Speed trap on longest straight"
    else:
        speed = pd.Series(top_speed)
        xlab = "Maximum recorded speed"

    fig, ax = plt.subplots(figsize=(10,8))
   
    for i in range(0, len(drivers)):
        style = fastf1.plotting.get_driver_style(drivers[i], get_custom_driver_styles('scatter'), session)
        ax.scatter(speed[i], fastest_laps['LapTime'][i], **style, label=drivers[i])

    fig.suptitle("Speed Versus Laptime \n"
                 f"{session.event['EventName']} {session.event.year} {session.session_info['Name']}",
                 fontsize='xx-large')
    
    ax.set_ylabel("Lap time")
    ax.set_xlabel(f"{xlab} (km/h)")
    ax.yaxis.set_major_formatter(tmpldelta.TimedeltaFormatter("%m:%s.%ms"))
    ax.grid(linewidth=0.5)

    for index, lap in fastest_laps.iterlaps():
        ax.text(speed[index]+0.2, lap['LapTime'], lap['Driver'])

    if save==True:
        plt.savefig(f"{F1DATABOT_PLOT_CACHE}/speed_versus_laptime {session.event['EventName']} {session.event.year} {session.session_info['Name']}.png", 
                    format='png', bbox_inches = 'tight')
        plt.clf()
    else:
        plt.show()


def plot_actions_on_track(session: fastf1.core.Session, driver: str,  
                          lap_number=None, save: bool = False) -> None:

    fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme = 'fastf1')

    # Get the laps and circuit info
    laps = session.laps
    circuit_info = session.get_circuit_info()

    # Extracting the laps

    if lap_number is None:
        laps_driver = laps.pick_drivers(driver).pick_fastest()
    else:
        laps_driver = laps.pick_drivers(driver).pick_laps(lap_number)  

    telemetry_driver_1 = laps_driver.get_car_data().add_distance()

    pos_driver_1 = laps_driver.get_pos_data()

    merged_driver_1 = telemetry_driver_1.merge_channels(pos_driver_1)

    # Assigning labels to what the drivers are currently doing 
    merged_driver_1.loc[(merged_driver_1['Brake'] > 0), 'CurrentAction'] = 1
    merged_driver_1.loc[(merged_driver_1['Brake'] > 0) & (merged_driver_1['Throttle'] > 0), 'CurrentAction'] = 2
    merged_driver_1.loc[(merged_driver_1['Throttle'] > 98) & (merged_driver_1['Brake'] == 0), 'CurrentAction'] = 4
    merged_driver_1.loc[(merged_driver_1['Brake'] == 0) & (merged_driver_1['Throttle'] <= 98), 'CurrentAction'] = 3

    x = np.array(merged_driver_1['X'].values)
    y = np.array(merged_driver_1['Y'].values)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    action = merged_driver_1['CurrentAction'].to_numpy()

    cmap = ListedColormap(['red', 'orange', 'grey', 'green'])
    lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N+1), cmap=cmap)
    lc_comp.set_array(action)
    lc_comp.set_linewidth(5)

    fig, ax = plt.subplots(figsize=(8,8))

    ax.add_collection(lc_comp)
    ax.set_xlim(x.min()-500, x.max()+500)
    ax.set_ylim(y.min()-500, y.max()+500)

    lap_time_string = strftimedelta(laps_driver['LapTime'], '%m:%s.%ms')

    ax.set_title(f"Throttle and Brake Visualisation\n"
                f"{session.event['EventName']} {session.event.year} {session.session_info['Name']} \n" 
                f"{laps_driver['Driver']}: {lap_time_string}",
                fontsize='xx-large')

    actions = ['Brake', 'Brake and Throttle', 'No Brake, Part Throttle', 'Full Throttle']
    action_colours = ['red', 'orange', 'grey', 'green']
    handles = []
    for index in range(0,4):
        handles.append(mpatches.Patch(color = action_colours[index], label = actions[index]))

    ax.legend(handles=handles, ncols = 1, loc = "upper left")

    def rotate(xy, *, angle):
        rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                            [-np.sin(angle), np.cos(angle)]])
        return np.matmul(xy, rot_mat)

    # Iterate over all corners.
    for _, corner in circuit_info.corners.iterrows():
        # Create a string from corner number and letter
        txt = f"{corner['Number']}{corner['Letter']}"

        # Convert the angle from degrees to radian.
        offset_angle = corner['Angle'] / 180 * np.pi

        # Rotate the offset vector so that it points sideways from the track.
        offset_x, offset_y = rotate([280,0], angle=offset_angle)

        # Add the offset to the position of the corner
        text_x = corner['X'] + offset_x
        text_y = corner['Y'] + offset_y

        # Draw a line from the track to this circle.
        #plt.plot([corner['X'], text_x], [corner['Y'], text_y], color='grey')

        # Draw a circle next to the track.
        plt.scatter(text_x, text_y, color='white', s=200)

        # Finally, print the corner number inside the circle.
        plt.text(text_x, text_y, txt,
                va='center_baseline', ha='center', size='large', color='black')
        
    plt.xticks([]),plt.yticks([])

    if lap_number == None:
        lap_text = "Fastest lap"
    else:
        lap_text = f"Lap {lap_number}"

    if save==True:
        plt.savefig(f"{F1DATABOT_PLOT_CACHE}/Actions on track {session.event['EventName']} {session.event.year} {session.session_info['Name']} {driver} {lap_text}.png", 
                    format='png', bbox_inches = 'tight')
        plt.clf()
    else:
        plt.show()


def plot_sector_performance(session: fastf1.core.Session, save:bool = False) -> None:

    laps = session.laps.pick_quicklaps(threshold=1.15)

    fastest_laps = uf.fastest_laps_in_session(laps)[['LapTime', 'Driver']]
    fastest_sectors = uf.fastest_sectors_in_session(laps)

    fastest_sectors['IdealLapTime'] = fastest_sectors['sector1'] + fastest_sectors['sector2'] + fastest_sectors['sector3']
    merged = fastest_sectors.merge(fastest_laps, how='outer', left_on='driver', right_on='Driver')
    merged['GapToIdeal'] = merged['LapTime'] - merged['IdealLapTime']

    s1 = merged[['driver', 'sector1']].sort_values(by='sector1').reset_index(drop=True)
    s1['sector1'] = s1['sector1'].dt.total_seconds()
    s1['sector1_diff'] = s1['sector1'] - s1['sector1'].min()

    s2 = merged[['driver', 'sector2']].sort_values(by='sector2').reset_index(drop=True)
    s2['sector2'] = s2['sector2'].dt.total_seconds()
    s2['sector2_diff'] = s2['sector2'] - s2['sector2'].min()

    s3 = merged[['driver', 'sector3']].sort_values(by='sector3').reset_index(drop=True)
    s3['sector3'] = s3['sector3'].dt.total_seconds()
    s3['sector3_diff'] = s3['sector3'] - s3['sector3'].min()

    fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')

    # plot the data
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, width_ratios=[1,1,1], figsize=(18,6))

    for idx, row in s1.iterrows():
        style = fastf1.plotting.get_driver_style(row['driver'], 
                                                 get_custom_driver_styles('bar'), 
                                                 session)
        ax1.barh(y=row['driver'], width=row['sector1_diff'], **style)
        
    ax1.invert_yaxis()
    ax1.grid(linewidth=0.5, alpha=0.5, axis='x')
    ax1.set_title(f"Sector 1", fontsize = 'large')
    ax1.set_xlabel("Difference to fastest (s)")
    ax1.set_ylabel("Driver")
    ax1.set_xlim(xmax=s1['sector1_diff'].max()+0.2)

    fastest_sector = s1['sector1'].min()
    # draw labels for lap times/deltas
    for index, lap in s1.iterrows():
        if lap['sector1'] == fastest_sector:
            ax1.text(0, 0.2, round(lap['sector1'], 3))
        else:
            ax1.text(lap['sector1_diff'], index+0.2, f"+{round(lap['sector1_diff'], 3)}s")

    for idx, row in s2.iterrows():
        style = fastf1.plotting.get_driver_style(row['driver'], 
                                                 get_custom_driver_styles('bar'), 
                                                 session)
        ax2.barh(y=row['driver'], width=row['sector2_diff'], **style)

    ax2.invert_yaxis()
    ax2.grid(linewidth=0.5, alpha=0.5, axis='x')
    ax2.set_title(f"Sector 2", fontsize = 'large')
    ax2.set_xlabel("Difference to fastest (s)")
    ax2.set_xlim(xmax=s2['sector2_diff'].max()+0.2)

    fastest_sector = s2['sector2'].min()
    # draw labels for lap times/deltas
    for index, lap in s2.iterrows():
        if lap['sector2'] == fastest_sector:
            ax2.text(0, 0.2, round(lap['sector2'], 3))
        else:
            ax2.text(lap['sector2_diff'], index+0.2, f"+{round(lap['sector2_diff'], 3)}s")

    for idx, row in s3.iterrows():
        style = fastf1.plotting.get_driver_style(row['driver'], 
                                                 get_custom_driver_styles('bar'), 
                                                 session)
        ax3.barh(y=row['driver'], width=row['sector3_diff'], **style)

    ax3.invert_yaxis()
    ax3.grid(linewidth=0.5, alpha=0.5, axis='x')
    ax3.set_title(f"Sector 3", fontsize = 'large')
    ax3.set_xlabel("Difference to fastest (s)")
    ax3.set_xlim(xmax=s3['sector3_diff'].max()+0.2)

    fastest_sector = s3['sector3'].min()
    # draw labels for lap times/deltas
    for index, lap in s3.iterrows():
        if lap['sector3'] == fastest_sector:
            ax3.text(0, 0.2, round(lap['sector3'], 3))
        else:
            ax3.text(lap['sector3_diff'], index+0.2, f"+{round(lap['sector3_diff'], 3)}s")

    plt.suptitle("Sector Performance \n"
                 f"{session.event['EventName']} {session.event.year} {session.session_info['Name']} \n",
                fontsize='xx-large', y=1.03)

    if save==True:
        plt.savefig(f"{F1DATABOT_PLOT_CACHE}/sector_performance {session.event['EventName']} {session.event.year} {session.session_info['Name']}.png", 
                    format='png', bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def plot_actual_vs_ideal_laptimes(session: fastf1.core.Session, save: bool = False) -> None:    

    fastf1.plotting.setup_mpl(misc_mpl_mods = False, color_scheme = 'fastf1')

    laps = session.laps.pick_quicklaps(threshold=1.15)
    drivers = np.unique(laps['Driver'])

    # Define a function to calculate the best theoretical laptime for a driver
    def best_theoretical_laptime(laps):
        
        best_s1 = pd.Series(laps['Sector1Time']).min()
        best_s2 = pd.Series(laps['Sector2Time']).min()
        best_s3 = pd.Series(laps['Sector3Time']).min()
        best_laptime = best_s1 + best_s2 + best_s3
        
        return best_laptime

    theoretical = []
    actual = []
    for driver in drivers:
        driver_laps = laps.pick_drivers(driver)
        if len(driver_laps.pick_quicklaps()) == 0:
            theoretical.append(pd.NaT)
            actual.append(pd.NaT)
        else:
            theoretical.append(best_theoretical_laptime(driver_laps.pick_quicklaps()))
            actual.append(driver_laps.pick_fastest(only_by_time=True)['LapTime'])


    laptime_df = pd.DataFrame({'driver': drivers, 'theoretical': theoretical, 'actual': actual})
    laptime_df = laptime_df.dropna(axis=0, how='any')
    laptime_df['theoretical'] = np.array([delta.total_seconds() for delta in laptime_df['theoretical']])
    laptime_df['actual'] = np.array([delta.total_seconds() for delta in laptime_df['actual']])
    laptime_df['difference'] = laptime_df['actual'] - laptime_df['theoretical']
    laptime_df = laptime_df.sort_values(by='actual', ascending=True).reset_index(drop=True)
    laptime_df['gap_to_fastest_theoretical'] = laptime_df['theoretical'] - laptime_df['theoretical'][0]
    laptime_df['gap_to_fastest_actual'] = laptime_df['actual'] - laptime_df['actual'][0]

    fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[1,1], figsize=(18,8))

    for idx, row in laptime_df.iterrows():
        style = fastf1.plotting.get_driver_style(row['driver'],
                                                 get_custom_driver_styles('bar'),
                                                 session)
        ax1.barh(row['driver'], row['gap_to_fastest_actual'], 
                **style, edgecolor = 'black')
        ax1.text(row['gap_to_fastest_actual']+0.1, idx+0.2, f"+{round(row['gap_to_fastest_actual'], 3)}s")
    
    ax1.invert_yaxis()
    ax1.set_xlabel("Gap to Fastest Actual Laptime (s)")
    ax1.set_ylabel("Driver")
    ax1.set_xlim(right = max(laptime_df['gap_to_fastest_actual']+0.3))
    ax1.set_title("Actual laptime order", fontsize='x-large')  

    laptime_df = laptime_df.sort_values(by='theoretical', ascending = True).reset_index(drop = True)

    for idx, row in laptime_df.iterrows():
        style = fastf1.plotting.get_driver_style(row['driver'],
                                                 get_custom_driver_styles('bar'),
                                                 session)
        ax2.barh(row['driver'], row['gap_to_fastest_theoretical'], 
                **style, edgecolor = 'black')
    
    ax2.barh(laptime_df['driver'], laptime_df['difference'], left = laptime_df['gap_to_fastest_theoretical'], 
            label = "Gap to actual laptime", color = 'mediumspringgreen', edgecolor = 'black', hatch = '///')
    ax2.invert_yaxis()
    ax2.set_xlabel("Gap to Fastest Ideal Laptime (s)")
    ax2.set_ylabel("Driver")
    ax2.set_xlim(right = max(laptime_df['gap_to_fastest_actual']+0.3))
    ax2.set_title("Ideal laptime order", fontsize='x-large')
    ax2.legend()

    for index, row in laptime_df.iterrows():
        ax2.text(row['gap_to_fastest_actual']+0.1, index+0.2, f"-{round(row['difference'], 3)}s",
                color='mediumspringgreen')

    plt.suptitle(f"Actual Versus Ideal Fastest Laptimes\n"
                f"{session.event['EventName']} {session.event.year} {session.session_info['Name']}",
                fontsize = 'xx-large')

    if save==True:
        plt.savefig(f"{F1DATABOT_PLOT_CACHE}/ideal_laptimes {session.event['EventName']} {session.event.year} {session.session_info['Name']}.png", 
                    format='png', bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


async def plot_position_changes(session: fastf1.core.Session, highlight: str = '', save: bool = False) -> None:

    fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')

    laps = uf.fix_positions(session.laps)
    drivers = np.unique(laps['Driver'])

    # set up the plot
    session_name = session.session_info['Name']
    if session_name == "Race":
        text_pos = -4
        axis_start = -5
        webpage_name = "starting-grid"
    else:
        text_pos = -2
        axis_start = -3
        webpage_name = "sprint-grid"

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_ylim([len(drivers)+0.5, 0.5])
    ax.set_yticks([1, 5, 10, 15, 20])
    ax.set_xlim([axis_start, np.max(laps['LapNumber'])+2])
    ax.set_xlabel('Lap')
    ax.set_ylabel('Position')

    grid_positions = f1web.process_starting_grid(f1web.get_session_results(session, webpage_name))

    if not highlight == '':
        highlight = highlight.split(",")

    # plot the positions for each driver
    for driver in drivers:
        driver_laps = laps.pick_drivers(driver).reset_index()

        if len(driver_laps) == 0:
            continue

        grid_pos = int(grid_positions.loc[grid_positions.Abbreviation == driver, "Pos"].iat[0])

        lap_series = pd.concat([pd.Series([0]), driver_laps['LapNumber']]).reset_index(drop=True)
        position_series = pd.concat([pd.Series([grid_pos]), driver_laps['Position']]).reset_index(drop=True)
        
        # color helper functions don't work for earlier years
        if session.event.year in [2023, 2024]:
            driver_style = fastf1.plotting.get_driver_style(driver, ["color", "linestyle"], session)
            ax.plot(lap_series, position_series,
                    label=driver, **driver_style, marker = 'o', markersize=4)
        else:
            ax.plot(lap_series, position_series, label=driver)

        ax.text(x=text_pos, y=position_series[0]+0.2, s=driver)

    plt.suptitle("Driver positions by lap \n"
                f"{session.event['EventName']} {session.event.year} {session.session_info['Name']}",
                fontsize='xx-large')
    
    if highlight == '':
        hightlight_text = ''
    else:
        hightlight_text = f" {'_'.join(highlight)}"

    if save==True:
        plt.savefig(f"{F1DATABOT_PLOT_CACHE}/position_changes {session.event['EventName']} {session.event.year} {session.session_info['Name']}{hightlight_text}.png", 
                    format='png', bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def plot_qualifying_lap_evolution(session, fastest_laps = True, save = False): 

    fastf1.plotting.setup_mpl(misc_mpl_mods = True, color_scheme = 'fastf1')

    # split quali into the sessions 
    q1, q2, q3 = session.laps.split_qualifying_sessions()

    if (fastest_laps == True):
        q1_fastest_laps = uf.fastest_laps_in_session(q1).pick_quicklaps()
        q2_fastest_laps = uf.fastest_laps_in_session(q2).pick_quicklaps()
        q3_fastest_laps = uf.fastest_laps_in_session(q3).pick_quicklaps()
        lap_text = "Fastest Laps"
    else:
        q1_fastest_laps = q1.pick_quicklaps()
        q2_fastest_laps = q2.pick_quicklaps()
        q3_fastest_laps = q3.pick_quicklaps()  
        lap_text = "All laps"

    # get the top 10
    top10 = np.unique(q3_fastest_laps['Driver'])

    # remove any drivers with times more than 10% below the session median
    for sesh in [q1_fastest_laps, q2_fastest_laps, q3_fastest_laps]:
        avg_time = np.median(sesh["LapTime"])
        mask = sesh['LapTime'] > 1.1*avg_time
        sesh = sesh.loc[~mask].reset_index()

    # Plot all the fastest laps over time to see track evolution
    all_fastest_laps = pd.concat([q1_fastest_laps, q2_fastest_laps, q3_fastest_laps])

    # calculate the median laptime in each session to see evolution
    # and remove outliers
    mean_laptimes = []
    mean_times = []
    for sesh in [q1_fastest_laps, q2_fastest_laps, q3_fastest_laps]:
        mean_laptimes.append(sesh.pick_drivers(top10)['LapTime'].mean())
        mean_times.append(sesh.pick_drivers(top10)['Time'].mean())

    fig, ax = plt.subplots(figsize=(10,8))

    drivers = np.unique(all_fastest_laps['Driver'])
    for driver in drivers:
        driver_laps = all_fastest_laps[all_fastest_laps['Driver']==driver]
        style = fastf1.plotting.get_driver_style(driver, get_custom_driver_styles('scatter'), session)
        ax.scatter(driver_laps['Time'], driver_laps['LapTime'], label = driver, **style)
        
    ax.plot(mean_times, mean_laptimes, "x--", label = "Top 10 Avg")
    ax.set_ylabel("Lap time")
    ax.set_xlabel("Session time")
    fig.legend(ncols = 1, loc = "upper left", bbox_to_anchor = (0.90, 0.89))
    ax.yaxis.set_major_formatter(TimedeltaFormatter("%m:%s:%ms"))
    ax.grid(linewidth=0.5, linestyle = "--")

    fig.suptitle(f"{session.session_info['Name']} Laptime Evolution - {lap_text}\n"
                 f"{session.event['EventName']} {session.event.year}", 
                 fontsize='xx-large')
    
    if save==True:
        plt.savefig(f"{F1DATABOT_PLOT_CACHE}/qualifying_lap_evolution {lap_text} {session.event['EventName']} {session.event.year} {session.session_info['Name']}.png", 
                    format='png', bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def plot_race_gaps(session, save=False, highlight: str = ''):

    fastf1.plotting.setup_mpl(color_scheme = 'fastf1', misc_mpl_mods = True)

    if not highlight == '':
        highlight = highlight.split(",")

    driver_laps = session.laps.loc[:,['Driver', 'DriverNumber', 'LapTime', 'LapNumber', 'Position', 'Time', 'LapStartTime', 'TrackStatus', 'PitInTime', 'PitOutTime']]
    
    # convert to seconds
    driver_laps.loc[:, "Time_seconds"] = np.array([d.total_seconds() for d in driver_laps.loc[:,"Time"]])
    driver_laps.loc[:, "LapStartTime_seconds"] = np.array([d.total_seconds() for d in driver_laps.loc[:,"LapStartTime"]])
    driver_laps.loc[:, "LapTime_seconds"] = np.array([d.total_seconds() for d in driver_laps.loc[:,"LapTime"]])

    # create alternative laptime as the difference of time
    driver_laps.loc[:,"DiffTime"] = driver_laps.loc[:,'Time_seconds'].groupby(driver_laps.loc[:,"Driver"]).diff()

    # DiffTime won't exist for the first lap so do Time - LapStartTime
    first_lap_mean_time = np.nanmean(driver_laps.pick_laps([1])['Time_seconds']) - np.nanmean(driver_laps.pick_laps([1])['LapStartTime_seconds'])

    # get top 10 drivers for the reference lap
    last_lap = np.max(driver_laps['LapNumber'].astype('int'))
    top10 = driver_laps.pick_laps(int(last_lap))
    top10 = top10.loc[top10['Position']<=10, "Driver"]
    top10_laps = driver_laps.pick_drivers(top10)

    # calculate the average pace for different track status
    clean_laps = uf.clean_laps(top10_laps).pick_wo_box()
    mean_pace = np.nanmean(clean_laps['LapTime_seconds'])

    laps_safety_car = uf.safety_car_laps(top10_laps)
    safety_car_laps = np.unique(laps_safety_car['LapNumber'].astype('int'))
    if(len(laps_safety_car)>0):
        mean_time_sf = laps_safety_car[['LapNumber', 'DiffTime']].groupby('LapNumber').mean()

    laps_vsc = uf.virtual_safety_car_laps(top10_laps)
    vsc_laps = np.unique(laps_vsc['LapNumber'].astype('int'))
    if(len(laps_vsc)>0):
        mean_time_vsc = laps_vsc[['LapNumber', 'DiffTime']].groupby('LapNumber').mean()

    laps_red_flag = uf.red_flag_laps(top10_laps)
    red_flag_laps = np.unique(laps_red_flag['LapNumber'].astype('int'))
    if(len(laps_red_flag)>0):
        mean_time_rf = laps_red_flag[['LapNumber', 'DiffTime']].groupby('LapNumber').mean()

    # build the laptimes for the reference car
    mean_time = []
    for lap in range(1, last_lap + 1):
        if lap==1:
            mean_time.append(first_lap_mean_time)
        else:
            if lap in red_flag_laps:
                mean_time.append(mean_time_rf.loc[mean_time_rf.index==lap, 'DiffTime'].iat[0])
            elif lap in safety_car_laps:
                mean_time.append(mean_time_sf.loc[mean_time_sf.index==lap, 'DiffTime'].iat[0])
            elif lap in vsc_laps:
                mean_time.append(mean_time_vsc.loc[mean_time_vsc.index==lap, 'DiffTime'].iat[0])
            else:
                mean_time.append(mean_pace)

    # add red flag stoppage time and amend the time for the restart lap
    if len(laps_red_flag) > 0:
        for red_flag in red_flag_laps.tolist():
            # add stoppage time to the red flag lap
            if red_flag + 1 <= last_lap:
                mean_time[int(red_flag)] = np.nanmean(top10_laps.pick_laps(red_flag+1)['DiffTime'])
            # set the time for the restart lap
            if red_flag + 2 <= last_lap:
                mean_time[int(red_flag) + 1] = np.nanmean(top10_laps.pick_laps(red_flag+2)['DiffTime'])

    # cumulate the laptime
    mean_time = np.cumsum(mean_time)

    # estimate the OLS equation: LapTime = a + b1*LapNumber + b2*LapNumber**2
    y, X = dmatrices(f'LapTime_seconds ~ LapNumber + I(LapNumber**2)', data=clean_laps, return_type='dataframe') 
    model = sm.OLS(y, X)
    results = model.fit()

    # use the estimated model to predict the trend change in laptime over the race
    # this will be mostly fuel burn
    lapnumber_array = np.arange(0, last_lap)
    predict_df = pd.DataFrame({'Intercept': 1, 'LapNumber': lapnumber_array, 'I(LapNumber ** 2)': lapnumber_array**2})
    ref_laps_adjustment = results.predict(predict_df) - results.params[results.params.index == "Intercept"].iat[0]
    ref_laps_adjustment = ref_laps_adjustment - ref_laps_adjustment.mean()

    mean_time = mean_time + ref_laps_adjustment

    # add the start time to turn the reference time into a time instead of a delta
    mean_time = mean_time + np.nanmin(top10_laps['LapStartTime_seconds'])
    mean_time = mean_time.set_axis(lapnumber_array+1)

    # pivot the driver data into columns and subtract the reference time
    pivoted_time = pd.pivot(driver_laps[['Driver', 'LapNumber', 'Time_seconds']], 
                            columns='Driver', index='LapNumber', values='Time_seconds')
    driver_gap = pivoted_time.sub(mean_time, axis = 0)

    # do the plot
    fig, ax = plt.subplots(figsize=(12,8))

    for driver in driver_gap.columns:

        if highlight == '': 
            alpha = 1
        else:
            if driver in highlight:
                alpha = 1
            else:
                alpha = 0.2

        style = fastf1.plotting.get_driver_style(driver, 
                                                 get_custom_driver_styles('line_marker'),
                                                 session)

        ax.plot(driver_gap.index, -driver_gap[driver], label=driver, 
                markersize = 5, alpha = alpha, **style)
        
    uf.add_special_lap_shading(session, ax)

    fig.legend(ncols = 1, loc = "upper left", bbox_to_anchor = (0.9, 0.88))
    fig.suptitle(f"Cumulative Race Gaps \n"
                 f"{session.event['EventName']} {session.event.year} {session.session_info['Name']}", 
                 fontsize='xx-large')
    ax.set_ylabel("Gap (s)")
    ax.set_xlabel("Lap")

    if highlight == '':
        hightlight_text = ''
    else:
        hightlight_text = f" {'_'.join(highlight)}"

    if save==True:
        plt.savefig(f"{F1DATABOT_PLOT_CACHE}/race_gaps {session.event['EventName']} {session.event.year} {session.session_info['Name']}{hightlight_text}.png", 
                    format='png', bbox_inches='tight')
        plt.clf()
    else:
        plt.show()
