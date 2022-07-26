from gwpy.timeseries import TimeSeries, TimeSeriesDict
from gwpy.time import to_gps

import numpy as np
from matplotlib import pyplot as plt

from typing import Any, Union

from datetime import datetime
from astropy.time import Time

import os
import jinja2 as j2

DEFAULT_CACHE_DIRECTORY = os.path.join(
    os.getenv("HOME"), 
    '.local', 
    'share', 
    'gwcluster'
)

def parse(arg: str) -> Any:
    '''
    tries to parse the value to an appropriate type, e.g. int if arg is an
    integer

    the current types that are attempted to convert to are int and float

    Parameters
    ----------
    arg : str
        argument to try and convert

    Returns
    -------
    Any
        the converted argument or the original arg if there was nothing to parse
    '''
    try:
        # first try to convert to int
        arg = int(arg)
    except ValueError:
        try:
            # if that doesn't work, try to convert to float
            arg = float(arg)
        except ValueError:
            # if none of the above worked, just return to the original arg
            pass

    return arg


def save_timeseries_to_cache(
    ts: TimeSeries,
    cache_directory: str = DEFAULT_CACHE_DIRECTORY
):
    '''
    saves a TimeSeries to local cache

    Parameters
    ----------
    ts : TimeSeries
        the TimeSeries to save (must have a channel name)
    cache_directory: str, default=DEFAULT_CACHE_DIRECTORY
        local path to cache directory
    '''
    # things saved under {cache_directory}/{channel_name}
    save_dir = os.path.join(cache_directory, ts.channel.name)
    if not os.path.exists(save_dir):
        # make directories if they don't exist
        os.makedirs(save_dir)

    # format of file is {start}_{end}.hdf5
    save_path = os.path.join(save_dir, f'{ts.span[0]}-{ts.span[1]}.hdf5')
    if not os.path.exists(save_path):
        # write file only if it doesn't already exist
        ts.write(save_path)

def timeseries_fetch(
    channel: str,
    start: Union[float, str, datetime, Time],
    end: Union[float, str, datetime, Time],
    cache_directory: str = DEFAULT_CACHE_DIRECTORY,
    cache_if_not_present: bool = True,
    *args: Any,
    **kwargs: Any,
) -> TimeSeries:
    '''
    wrapper for TimeSeries.fetch that checks local cache before fetching from
    remote host

    Parameters
    ----------
    channel : str
        name of TimeSeries' channel
    start : str
        start time (anything parsable by to_gps)
    end : str
        end time (anything parsable by to_gps)
    cache_directory : str
        path to directory in which to cache data
    cache_if_not_present : bool, default=True
        if true, will store data locally after fetching from remote host
    *args : Any
        any args acceptable by TimeSeries.fetch
    **kwargs : Any
        any kwargs acceptable by TimeSeries.fetch
    '''
    # paths where things will be saved
    download_dir = os.path.join(cache_directory, channel)

    # flag indicating if we found data locally
    found = False
    if os.path.exists(download_dir):
        # proceed only if channel directory exists
        for tsfile in os.listdir(download_dir):
            # for each file in the channel directory...
            # files are named by {start}_{end}.hdf5 where both start and
            #   end are gps iso formatted times
            tsfile_start, tsfile_end = os.path.splitext(tsfile)[0].split('-')
            if to_gps(start) >= to_gps(tsfile_start) and to_gps(end) <= to_gps(tsfile_end):
                # if the pass start, end times fall within one of the file's
                #   start, end times use it
                found = True
                print(f'found {channel} in local cache')
                # read in TimeSeries from local file
                ts = TimeSeries.read(
                    os.path.join(download_dir, tsfile)
                # then crop it to start, end
                ).crop(to_gps(start), to_gps(end))

    if not found:
        # if TimeSeries not found in local cache, fetch from remote
        ts = TimeSeries.fetch(channel, start, end, *args, **kwargs)
        if cache_if_not_present:
            # if not present and the user wants to cache, save downloaded data
            save_timeseries_to_cache(ts, cache_directory)

    # lest we forget to return the TimeSeries!
    return ts

def timeseriesdict_from_list(tss: list[TimeSeries]) -> TimeSeriesDict:
    '''
    convert a TimeSeries list to a TimeSeriesDict

    Parameters
    ----------
    tss: list[TimeSeries]
        list of TimeSeries; all items must have a channel name

    Returns
    -------
    TimeSeriesDict
        maps a channel name to the corresponding TimeSeries
    '''
    tsd = TimeSeriesDict()

    for ts in tss:
        # key is channel name, value is TimeSeries itself
        tsd.update(TimeSeriesDict.fromkeys([ts.channel.name], ts))

    return tsd

def states_plot(cluster_labels: np.ndarray):
    '''indicates the sequence of states (the cluster that the current 
    vector is apart of) over the sampled time'''
    # generate axes
    fig, ax = plt.subplots()

    # plot the states (cluster labels) vs. time
    ax.plot(cluster_labels, drawstyle='steps-post')

    # adjust y axis ticks to only be on labels
    ax.set_yticks(list(set(cluster_labels)))

    # set axes labels with informative names
    ax.set_xlabel(f'Time [minutes]')
    ax.set_ylabel('state')

    return fig


def parse_blrms_channel_name(channel: str) -> dict[str, str]:
    '''
    parses blrms seismometer channel name into components

    i.e. {ifo}:{system}-RMS_{signal}_{direction}_{low_freq}_{high_freq}.mean
    --> {'ifo': ifo, 'system': system, ... }

    Parameters
    ----------
    channel : str
        the channel name

    Returns
    -------
    dict[str,str]
        dictionary of the components of the blrms channel name
    '''
    def parse_frequency(freq: str) -> str:
        if not 'p' in freq:
            return freq
        return '.'.join(freq.split('p'))

    colon_split = channel.split(':')
    ifo = colon_split[0]

    dash_split = colon_split[1].split('-')
    system = dash_split[0]

    underscore_split = dash_split[1].split('_')
    signal = underscore_split[1]
    direction = underscore_split[2]
    low_freq = parse_frequency(underscore_split[3])
    high_freq = parse_frequency(underscore_split[4].split('.')[0])

    return {
        'ifo': ifo,
        'system': system,
        'signal': signal,
        'direction': direction,
        'low_freq': low_freq,
        'high_freq': high_freq,
    }
