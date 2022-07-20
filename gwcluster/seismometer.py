'''
for clustering time series data from seismometers
'''

from sklearn.cluster import KMeans, DBSCAN

from gwpy.timeseries import TimeSeriesDict
from gwpy.frequencyseries import FrequencySeries

import numpy as np

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
        self.clustering.fit(self.vectors)
        
        # get the resulting labels from the clustering
        self.labels = self.clustering.labels_

        # get the number of clustering based on the number of distinct labels
        self.n_clusters = len({label for label in self.labels if label >= 0})

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
        vectors = self.vectors
        # assume to start that the closest index is the first vector in vectors
        closest_index = 0

        for i in range(len(vectors)):
            if np.linalg.norm(
                vector - vectors[i]
            ) < np.linalg.norm(
                vector - vectors[closest_index]
            ):
                # if vectors[i] is closer to vector than vectors[closest_index]
                #   then update closest_index to i
                closest_index = i

        return closest_index

    def psds_of_centers(
        self, 
        psd_length: float = 60,
    ) -> list[list[FrequencySeries]]:
        '''
        computes the power spectral density (PSD) of each center in self.centers

        Parameters
        ----------
        psd_length : float, default=60
            number of seconds from the time that the center occurred for which
            to compute the psd
        '''
        times = self.times
        centers = self.centers
        times_of_centers = [
            # find the vector closest to center and use the corresponding time
            #   for that vector
            times[self.closest_index(center)].value
            for center in centers
        ]

        # copy the raw TimeSeriesDict n_clusters times (1 time for each center)
        #   since crop overwrites the TimeSeriesDict
        raw_copies = [self.raw.copy() for _ in range(self.n_clusters)]

        raw_cropped_to_centers = [
            # crop the raw data to a psd_length window around the point in
            #   time that center occurred
            raw_copy.crop(
                start=times_of_centers[i],
                end=times_of_centers[i] + psd_length,
                copy=True
            )
            for i, raw_copy in enumerate(raw_copies)
        ]

        return [
            # compute the the psd for each cropped raw TimeSeries
            [
                cropped_ts.psd().crop(0.03, 30)
                for cropped_ts in raw_cropped_to_center.values()
            ] 
            for raw_cropped_to_center in raw_cropped_to_centers
        ]
