import typer

from .seismometer import ClusteredSeismicData
from .config import get_config, clustering_default
from gwpy.timeseries import TimeSeriesDict, TimeSeries
from sklearn.cluster import KMeans
from gwpy.plot import Plot
from gwpy.time import to_gps
import os
import yaml
from typing import Optional, Any

# the cli
app = typer.Typer()

def parse(arg: str) -> Any:
    '''
    tries to parse the value to an appropriate type, e.g. int if arg is an
    integer
    '''
    try:
        arg = int(arg)
    except ValueError:
        try:
            arg = float(arg)
        except ValueError:
            pass

    return arg

def timeseries_fetch(
    channel: str,
    start: str,
    end: str,
    host: str,
    port: int,
    download_directory: str,
    **kwargs
) -> TimeSeries:
    # paths where things will be saved
    download_dir = os.path.join(download_directory, channel)

    if os.path.exists(download_dir):
        for tsfile in os.listdir(download_dir):
            tsfile_start, tsfile_end = os.path.splitext(tsfile)[0].split('-')
            if to_gps(start) >= to_gps(tsfile_start) and to_gps(end) <= to_gps(tsfile_end):
                print(f'found {channel} locally')
                return TimeSeries.read(os.path.join(download_dir, tsfile)).crop(to_gps(start), to_gps(end))

    return TimeSeries.fetch(channel, start, end, host, port, verbose=True)

def timeseriesdict_from_list(timeseriess: list[TimeSeries]) -> TimeSeriesDict:
    tsd = TimeSeriesDict()

    for ts in timeseriess:
        tsd.update(TimeSeriesDict.fromkeys([ts.channel.name], ts))

    return tsd


default_clustering = clustering_default('seismometer')

@app.command()
def seismometer(
    clustering: str = typer.Option(default_clustering),
    clustering_kwargs: str = typer.Option(None),
    ifo: str = typer.Option(get_config('seismometer', 'ifo')),
    system: str = typer.Option(get_config('seismometer', 'system')),
    signal: str = typer.Option(get_config('seismometer', 'signal')),
    start: str = typer.Option(get_config('seismometer', 'start')),
    end: str = typer.Option(get_config('seismometer', 'end')),
    host: str = typer.Option(get_config('seismometer', 'host')),
    port: int = typer.Option(get_config('seismometer', 'port')),
    download_directory: str = typer.Option(get_config('download_directory')),
    output_directory: str = typer.Option(get_config('output_directory'))
):
    '''
    performs clustering on seismometer data
    '''
    # check all necessary parameters are set
    for param in ['clustering', 'ifo', 'system', 'signal', 'start', 'end', 'host', 'port']:
        if not vars()[param]:
            print(f'Error: "{param}" not specified on command line or in config file')
            raise typer.Exit(1)

    # parse clustering kwargs
    if not clustering_kwargs:
        parsed_clustering_kwargs = get_config('seismometer', 'clustering', default_clustering)
        parsed_clustering_kwargs = parsed_clustering_kwargs if parsed_clustering_kwargs else {}
    else:
        # if it's a string, assume it was passed on the command line in
        #   "key1=value1,key2=value2,..." format
        parsed_clustering_kwargs = {
            pair.split('=')[0]: parse(pair.split('=')[1])
            for pair in clustering_kwargs.split(',')
        }

    # directional movement of seismometer
    directions = ['X', 'Y', 'Z']

    # channels with raw data
    raw_channels = [
        f'{ifo}:{system}-SEIS_{signal}_{dir}_OUT_DQ'
        for dir in directions
    ]

    # channels with blrms data
    blrms_channels = [
        f'{ifo}:{system}-RMS_{signal}_{dir}_{band}.mean'
        for band in [
            '0p03_0p1',
            '0p1_0p3',
            '0p3_1',
            '1_3',
            '3_10',
            '10_30',
        ]
        for dir in directions
    ]

    # other parameters to pass to TimeSeriesDict
    non_channel_params = {
        "host": host,
        "port": port,
        "start": start,
        "end": end,
        "download_directory": download_directory
    }

    csd = ClusteredSeismicData(
        raw=timeseriesdict_from_list([timeseries_fetch( channel, **non_channel_params) for channel in raw_channels]),
        #raw=TimeSeriesDict.get(
        #    channels=raw_channels,
        #    **non_channel_params
        #),
        blrms=timeseriesdict_from_list([timeseries_fetch( channel, **non_channel_params) for channel in blrms_channels]),
        #blrms=TimeSeriesDict.get(
        #    channels=blrms_channels,
        #    **non_channel_params
        #),
        clustering=eval(f'{clustering}(**{parsed_clustering_kwargs})')
    )

    full_output_dir = os.path.join(output_directory, ifo, system)
    if not os.path.exists:
        os.makedirs(full_output_dir)

    for i, psds in enumerate(csd.psds_of_centers(psd_length=600)):
        plot = Plot(*psds, xscale='log', yscale='log', ylabel='Velocity')
        plot.save(os.path.join(full_output_dir, f'center{i}-psd.png'))

@app.command()
def download(
    channel: str,
    start: str = typer.Option(get_config('seismometer', 'start')),
    end: str = typer.Option(get_config('seismometer', 'end')),
    host: str = typer.Option(get_config('seismometer', 'host')),
    port: int = typer.Option(get_config('seismometer', 'port')),
    download_directory: str = typer.Option(get_config('download_directory')),
):
    '''
    downloads TimeSeries from specified channel and saves it to local disk

    the data is stored at {download directory}/{channel}/{start}-{end}
    '''
    # fetch time series from remote
    ts = TimeSeries.fetch(channel, start, end, host, port, verbose=True)

    # paths where things will be saved
    download_dir = os.path.join(download_directory, channel)
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    download_path = os.path.join(download_dir, f'{ts.span[0]}-{ts.span[1]}.hdf5')
    if not os.path.exists(download_path):
        ts.write(download_path)
