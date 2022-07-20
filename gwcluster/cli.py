import typer
from seismometer import ClusteredSeismicData
from gwpy.timeseries import TimeSeriesDict
from sklearn.cluster import KMeans
from gwpy.plot import Plot
import os
import yaml
from typing import Optional, Any

CONFIG_FILE_PATH = f'{os.getenv("HOME")}/.gwcluster.yml'

# the cli
app = typer.Typer()

# user config settings
if os.path.exists(CONFIG_FILE_PATH):
    with open(CONFIG_FILE_PATH, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
else:
    config = None

def default(
    param: Optional[Any],
    parent_key: str,
    config_key: str
) -> Any:
    '''
    defaults parameter to value in config if parameter is not set

    Parameters
    ----------
    param : Any
        variable that could None or could have a value
    parent_key : str
        key to dictionary with relevant configs (e.g. 'seismometer')
    config_key : str
        key under parent_key
        
    '''
    if not param:
        try:
            param = config[parent_key][config_key]
        except (TypeError, KeyError):
            raise Exception(f'{parent_key}.{config_key} not found in config file at {CONFIG_FILE_PATH}')
    return param

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


@app.command()
def seismometer(
    clustering: Optional[str] = None,
    clustering_kwargs: Optional[str] = None,
    ifo: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
):
    '''
    performs clustering on seismometer data

    Parameters
    ----------
    clustering : str, defaults to value in config
        clustering algorithm to use
    clustering_kwargs : str, defaults to value in config
        kwargs to pass to clustering algorithm, formatted as "key1=value1,key2=value2,..."
    ifo : str, defaults to value in config
        ifo of channels to grab seismometer data from
    start : str, default to value in config
        start time (must be gps parsable)
    end : str, default to value in config
        end time (must be gps parsable)
    host : str, default to value in config
        host to grab channel data from
    port : int, default to value in config
        port on host
    '''
    # default parameters from config file
    ifo = default(ifo, 'seismometer', 'ifo')
    start = default(start, 'seismometer', 'start')
    end = default(end, 'seismometer', 'end')
    host = default(host, 'seismometer', 'host')
    port = default(port, 'seismometer', 'port')

    # default clustering algorithm
    if clustering:
        parsed_clustering_kwargs = {
            pair.split('=')[0]: parse(pair.split('=')[1])
            for pair in clustering_kwargs.split(',')
        }
    else:
        clustering_from_config = config['seismometer']['clustering']
        clustering, parsed_clustering_kwargs = next(iter(clustering_from_config.items()))


    # directional movement of seismometer
    directions = ['X', 'Y', 'Z']

    # channels with raw data
    raw_channels = [
        f'{ifo}:PEM-SEIS_BS_{dir}_OUT_DQ'
        for dir in directions
    ]

    # channels with blrms data
    blrms_channels = [
        f'{ifo}:PEM-RMS_BS_{dir}_{band}.mean'
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
        "verbose": True
    }

    csd = ClusteredSeismicData(
        raw=TimeSeriesDict.get(
            channels=raw_channels,
            **non_channel_params
        ),
        blrms=TimeSeriesDict.get(
            channels=blrms_channels,
            **non_channel_params
        ),
        clustering=eval(f'{clustering}(**{parsed_clustering_kwargs})')
    )

    for psds in csd.psds_of_centers(psd_length=600):
        plot = Plot(*psds, xscale='log', yscale='log')
        plot.show()

if __name__ == '__main__':
    app()
