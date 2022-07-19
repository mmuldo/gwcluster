import typer
from seismometer import ClusteredSeismicData
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from sklearn.cluster import KMeans
from gwpy.plot import Plot

app = typer.Typer()

@app.command()
def seismometer(
    ifo: str,
    start: str,
    end: str,
    host: str = '',
    port: int = 0
):
    dirs = ['X', 'Y', 'Z']

    raw_channels = [
        f'{ifo}:PEM-SEIS_BS_{dir}_OUT_DQ'
        for dir in dirs
    ]

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
        for dir in dirs
    ]
    non_channel_params = {
        "host": host,
        "port": port,
        "start": start,
        "end": end,
        "verbose": True
    }

    csd = ClusteredSeismicData(
        raw=TimeSeriesDict.fetch(
            channels=raw_channels,
            **non_channel_params
        ),
        blrms=TimeSeriesDict.fetch(
            channels=blrms_channels,
            **non_channel_params
        ),
        clusters=KMeans(n_clusters=5)
    )

    csd.clusters.fit(csd.vectors())
    for psds in csd.psds_of_centers():
        plot = Plot(*psds, xscale='log', yscale='log')
        plot.show()

if __name__ == '__main__':
    app()
