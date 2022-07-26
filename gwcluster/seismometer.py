'''
for clustering time series data from seismometers
'''

from gwpy.plot.plot import Plot
from gwpy.time import from_gps

from gwpy.timeseries import TimeSeriesDict
from gwpy.frequencyseries import FrequencySeries

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, LogNorm

import numpy as np
from . import util

from typing import Any


class ClusteredSeismicData:
    '''
    clusters data from a seismometer using a set of raw data streams and a set 
    of bandlimited root-mean-squared (BLRMS) data streams

    Parameters
    ----------
    raw : TimeSeriesDict
        set (e.g. x-, y-, and z-coordinates) of raw data from the seismometer
    blrms : TimeSeriesDict
        set (e.g. multiple frequency bands for each coordinate) of BLRMS data
        from the seismometer
    clustering : Any
        clustering algorithm from scikit-learn to cluster the data

    Attributes
    ----------
    raw : TimeSeriesDict
        set (e.g. x-, y-, and z-coordinates) of raw data from the seismometer
    blrms : TimeSeriesDict
        set (e.g. multiple frequency bands for each coordinate) of BLRMS data
        from the seismometer
    clustering : Any
        clustering algorithm from scikit-learn to cluster the data
    vectors : np.ndarray of shape (number of samples, number of blrms time series)
        vectorized blrms data
    times : array-like of shape (number of samples)
        list of gps-formatted times corresponding to each sample
    labels : np.ndarry of shape (number of samples)
        indicates the cluster that each vector belongs to
    n_clusters : int
        number of clusters
    centers : np.ndarray of shape (n_clusters, number of blrms time series)
        the average vector of each cluster
    '''
    def __init__(
        self,
        raw: TimeSeriesDict,
        blrms: TimeSeriesDict,
        clustering: Any
    ):
        self.raw = raw
        self.blrms = blrms
        self.clustering = clustering

        # reduce self.blrms to vectors
        self.vectors = np.array(
            list(self.blrms.values())
        ).T

        try:
            # get the times list from the times attribute of the first
            #   TimeSeries in self.raw
            self.times = next(iter(raw.values())).times
        except StopIteration:
            # all of the TimeSeries in self.raw should contain the times
            #   attribute, so this should never be hit
            assert False

        # fit the vector data using the specified clustering algorithm
        print('clustering data...')
        self.clustering.fit(self.vectors)
        print('done')
        
        # get the resulting labels from the clustering
        self.labels = self.clustering.labels_

        # get the number of clustering based on the number of distinct labels
        self.n_clusters = len({label for label in self.labels if label >= 0})

        print('finding centers of clusters...')
        # organize vectors by cluster label
        clusters = [[] for _ in range(self.n_clusters)]
        for i in range(len(self.vectors)):
            if self.labels[i] not in range(self.n_clusters):
                # skip data that isn't in a cluster
                continue
    
            # put the vector in its proper cluster
            clusters[self.labels[i]].append(self.vectors[i])

        # get the centers by reducing each cluster to its average vector
        self.centers = np.array([
            np.mean(np.array(clust), axis=0)
            for clust in clusters
        ])
        print('done')

    def closest_index(
        self,
        vector: np.ndarray
    ) -> int:
        '''
        gets the index in self.vectors of the closest vector to the passed vector
    
        Parameters
        ----------
        vector : np.ndarray of shape (1, number of blrms time series)
            a vector
    
        Returns
        -------
        int
            index of closest vector in self.vectors to the passed vector
        '''
        distances = np.linalg.norm(self.vectors - vector, axis=1)
        return distances.argmin()

    def psds_of_centers(
        self, 
        psd_length: float = 600,
    ) -> list[list[FrequencySeries]]:
        '''
        computes the power spectral density (PSD) of each center in self.centers

        Parameters
        ----------
        psd_length : float, default=60
            number of seconds from the time that the center occurred for which
            to compute the psd

        Returns
        -------
        list[list[FrequencySeries]]
            list where each element is a list of psds associated with the
            given center
        '''
        centers = self.centers
        indices_of_centers = [
            # find the vector closest to center and use the corresponding index
            #   for that vector
            self.closest_index(center)
            for center in centers
        ] 
        # convert raw TimeSeriesDict to a list of TimeSeries
        raw_tss = list(self.raw.values())

        # crop the list of TimeSeries to each center
        crops = []
        for i in indices_of_centers:
            # crops for the current center
            cropped_tss = []
            for raw_ts in raw_tss:
                # convert psd_length to length in number of array indices
                n_samples = int(psd_length / raw_ts.dt.value)
                if i + n_samples < raw_ts.size:
                    # if slice doesn't exceed length of TimeSeries,
                    #   crop n_samples starting at center's index
                    cropped_tss.append(raw_ts[i: i + n_samples])
                else:
                    # otherwise, need to crop backwards so that we don't
                    #   index out of range
                    cropped_tss.append(raw_ts[i - n_samples: i])
            crops.append(cropped_tss)


        return [
            # compute the the psd for each cropped raw TimeSeries
            [
                cropped_ts.psd(fftlength=60, method='welch').crop(10**(-3/2), 10**(3/2))
                for cropped_ts in crops[i]
            ]
            for i in range(self.n_clusters)
        ]

    def center_plots(self) -> list[Plot]:
        '''
        returns a list of plots where each plot is a visual representation
        of a cluster center

        Returns
        -------
        list[Plot]
            list of plots of centers
        '''
        kwargs = {
            'xscale': 'log',
            'yscale': 'log',
            'ylabel': r'Velocity [$(\mu m / s)/ \sqrt{Hz}$]'
        }

        return [
            Plot(*psds, **kwargs)
            for psds in self.psds_of_centers()
        ]

    def vectors_plot(self) -> Figure:
        '''returns a pcolormesh of coordinates (of the vectors) vs. time'''
        # x axis is fine as is (seconds starting from 0)
    
        ## y axis demarcates the frequency bands
        ## given n frequency bands, we need n+1 frequency markings (an extra
        ##   one for the highest frequency)
        #y = list(range(len(self.centers[0])+1))
        ## the actual labels are the frequencies themselves
        parsed_channels = [
            util.parse_blrms_channel_name(channel)
            for channel in list(self.blrms.keys())
        ]
        ylabels = [
            f'{parsed_channel["direction"]}: '
            f'{parsed_channel["low_freq"]} - {parsed_channel["high_freq"]} Hz'
            for parsed_channel in parsed_channels
        ]
    
        # generate figure and axes
        fig, ax = plt.subplots()
    
        # generate pseudocolor plot
        # need to transpose vectors, otherwise pcolormesh plots them
        #   transposed for some reason
        vectors = np.flip(self.vectors.T, axis=0)
        pcm = ax.pcolormesh(
            vectors, 
            norm=LogNorm(vmin=vectors.min(), vmax=vectors.max()),
        )
    
        # adjust axes tick labels
        ax.set_yticks(list(range(self.vectors.shape[1])))
        ax.set_yticklabels(ylabels)
    
        # set axes labels with informative names
        ax.set_xlabel(f'Time [seconds] after {self.times[0]}' )
        ax.set_ylabel('channels')
    
        # generate colorbar
        fig.colorbar(
            pcm, 
            ax=ax,
            label=r'Velocity [$(\mu m / s)/ \sqrt{Hz}$]',
        )

        return fig
    
def raw_channels(
    ifo: str,
    system: str,
    signal: str
) -> list[str]:
    '''
    returns a list of channels relevant to the raw data associated with the
    passed in paremeters

    note that channels containing raw seismomter data are fomatted as:
        {ifo}:{system}-SEIS_{signal}_{direction}_OUT_DQ

    Parameters
    ----------
    ifo : str
        the channel's ifo (e.g. C1)
    system : str
        the channel's system (e.g. PEM)
    signal : str
        the channel's signal (e.g. BS)

    Returns
    -------
    list[str]
        all raw data channels associated with passed parameters
    '''
    return [
        f'{ifo}:{system}-SEIS_{signal}_{direction}_OUT_DQ'
        # return a channel for each 3D-coordinate
        for direction in ['X', 'Y', 'Z']
    ]

def blrms_channels(
    ifo: str,
    system: str,
    signal: str
) -> list[str]:
    '''
    returns a list of channels relevant to the BLRMS (bandlimited 
    root-mean-squared) data associated with the passed in paremeters

    note that channels containing blrms seismomter data are fomatted as:
        {ifo}:{system}-RMS_{signal}_{direction}_{low_freq}_{high_freq}.mean

    Parameters
    ----------
    ifo : str
        the channel's ifo (e.g. C1)
    system : str
        the channel's system (e.g. PEM)
    signal : str
        the channel's signal (e.g. BS)

    Returns
    -------
    list[str]
        all raw data channels associated with passed parameters
    '''
    # in Hz
    cutoffs = [
        # 0.03
        '0p03',
        # 0.1
        '0p1',
        # 0.3
        '0p3',
        # 1
        '1',
        # 3
        '3',
        # 10
        '10',
        # 30
        '30',
    ]

    frequency_bands = [
        (cutoffs[i], cutoffs[i+1])
        for i in range(len(cutoffs) - 1)
    ]

    return [
        f'{ifo}:{system}-RMS_{signal}_{direction}_{low_freq}_{high_freq}.mean'
        # return a channel for each 3D-coordinate
        for direction in ['X', 'Y', 'Z']
        # return a channel for each frequency band
        for low_freq, high_freq in frequency_bands
    ]
