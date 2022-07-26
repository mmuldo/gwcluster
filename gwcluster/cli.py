'''
cli for gwcluster

for help:
    python -m gwcluster --help
'''

import typer

from sklearn.cluster import KMeans, OPTICS

from gwpy.timeseries import TimeSeriesDict, TimeSeries
from gwpy.plot import Plot
from gwpy.time import to_gps

import os
import jinja2 as j2
from corner import corner

from .seismometer import ClusteredSeismicData, raw_channels, blrms_channels
from .config import Config
from . import util

# the cli
app = typer.Typer()

# config file
config = Config()

# jinja2 environment (for templates)
env = j2.Environment(
    loader=j2.FileSystemLoader(os.path.join(os.path.dirname(__file__), 'templates')),
    trim_blocks=True,
    lstrip_blocks=True
)

@app.command()
def seismometer(
    clustering: str = typer.Option(config.clustering('seismometer')),
    clustering_kwargs: str = typer.Option(None),
    ifo: str = typer.Option(config.get('seismometer', 'ifo')),
    system: str = typer.Option(config.get('seismometer', 'system')),
    signal: str = typer.Option(config.get('seismometer', 'signal')),
    start: str = typer.Option(config.get('seismometer', 'start')),
    end: str = typer.Option(config.get('seismometer', 'end')),
    host: str = typer.Option(config.get('seismometer', 'host')),
    port: int = typer.Option(config.get('seismometer', 'port')),
    output: str = typer.Option(config.get('output')),
    cache_data: bool = typer.Option(config.get('cache_data')),
):
    '''
    performs clustering on seismometer data

    all parameters default to their respective values in ~/.gwcluster.yml
    '''
    print('***********GWCLUSTER**********')
    # check all necessary parameters are set
    for param in ['clustering', 'ifo', 'system', 'signal', 'start', 'end', 'host', 'port', 'output']:
        if not vars()[param]:
            print(f'Error: "{param}" not specified on command line or in config file')
            raise typer.Exit(1)
        print(f'{param}: {vars()[param]}')
    print('******************************')

    # parse clustering kwargs
    if not clustering_kwargs:
        parsed_clustering_kwargs = config.get('seismometer', 'clustering', clustering)
        parsed_clustering_kwargs = parsed_clustering_kwargs if parsed_clustering_kwargs else {}
    else:
        # if it's a string, assume it was passed on the command line in
        #   "key1=value1,key2=value2,..." format
        parsed_clustering_kwargs = {
            pair.split('=')[0]: util.parse(pair.split('=')[1])
            for pair in clustering_kwargs.split(',')
        }

    # other parameters to pass to TimeSeriesDict
    non_channel_params = {
        "host": host,
        "port": port,
        "start": start,
        "end": end,
        "verbose": True,
        "cache_if_not_present": cache_data if cache_data is not None else False
    }

    # retrieve TimeSeriesDict's from util functions
    raw, blrms = (
        util.timeseriesdict_from_list(
            util.timeseries_fetch(channel, **non_channel_params)
            for channel in channels(ifo, system, signal)
        )
        # note that raw_channels() and blrms_channels are functions
        for channels in [raw_channels, blrms_channels]
    )

    # the clustered object
    csd = ClusteredSeismicData(
        raw=raw,
        blrms=blrms,
        clustering=eval(f'{clustering}(**{parsed_clustering_kwargs})')
    )

    # path where results will be saved
    full_output_dir = os.path.join(
        output,
        ifo,
        system,
        clustering,
        f'{to_gps(start)}_{to_gps(end)}',
    )
    if not os.path.exists(full_output_dir):
        # make paths if they don't already exist
        os.makedirs(full_output_dir)

    # save centers plots
    for i, plot in enumerate(csd.center_plots()):
        print(f'saving center plot for cluster {i}...')
        plot.save(os.path.join(full_output_dir, f'center{i}-psd.png'))
        print('done')

    # save other plots
    for filename, figure in {'vectors.png': csd.vectors_plot(), 'corner.png': corner(csd.vectors)}.items():
        print(f'saving {filename}...')
        figure.savefig(os.path.join(full_output_dir, filename), bbox_inches='tight')
        print('done')

    # generate markdown summary
    print(f'generating markdown summary...')
    markdown = env.get_template('seismometer-summary.md.j2').render(
        ifo=ifo,
        system=system,
        signal=signal,
        start=start,
        end=end,
        clustering=clustering,
        clustering_kwargs=parsed_clustering_kwargs,
        n_clusters=csd.n_clusters
    )
    with open(os.path.join(
        full_output_dir,
        # filename is the kwargs (separated by dashes) passed to clustering algorithm
        '-'.join([
            f'{k}={v}' 
            for k, v in parsed_clustering_kwargs.items()
        ]) + '.md',
    ), 'w') as f:
        f.write(markdown)
    print('done')


@app.command()
def download(
    channel: str,
    start: str = typer.Option(config.get('seismometer', 'start')),
    end: str = typer.Option(config.get('seismometer', 'end')),
    host: str = typer.Option(config.get('seismometer', 'host')),
    port: int = typer.Option(config.get('seismometer', 'port')),
    download_directory: str = typer.Option(config.get('download_directory')),
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
