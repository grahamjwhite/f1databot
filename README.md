# f1databot

## Features

`f1databot` is a Discord bot that can fetch data for Formula One races (using the [fastf1](https://github.com/theOehrly/Fast-F1) package for python) and create plots. Users request plots from the bot using interactions (a.k.a slash commands) in Discord and the bot responds with the requested plot.

The plots that can currently be produced are:

- A map of a track
- The weather during a session (temp, rainfall)
- The tyre usage (compound type and length of stint) by each driver during a session
- The fastest laps completed by all drivers in a session
- The fastest laps completed by all drivers in each qualifying session
- The fastest laps versus fastest 'ideal' laps for all drivers in a session
- The fastest lap time versus fastest top speed for each driver in a session
- The ranked sector times for all drivers and each sector in a session
- The lap time evolution over a qualifying session
- Telemetry comparison for two drivers over one lap (speed, braking, gear, DRS, time delta)
- Cornering comparison for two drivers for a single corner of a lap
- The race positions for all drivers for all laps over a race
- The race gaps for all drivers for all laps over a race compared to a reference time


## Usage

1. Clone the repository: `git clone https://github.com/grahamjwhite/f1databot.git`
2. Create a Discord bot using the [Discord developer portal](https://discord.com/developers/). There are good guides around for doing this. 
3. Set the environment variables required in `constants.py`
4. Run the bot by running the `bot.py` file in python: `python bot.py`

## Supporting the project

If you find this project useful, please consider supporting it by:

- Acknowledging this repo if you use it for your own project
- [Buying me a coffee](https://www.buymeacoffee.com/grahamjwhite)
- Starring the repository on GitHub
- Reporting issues and suggesting features
- Contributing code via pull requests

