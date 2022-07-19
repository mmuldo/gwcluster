from astropy.units.quantity import Quantity
from astropy.time import Time
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from gwpy.frequencyseries import FrequencySeries
from gwpy.types.index import Index
from gwpy.time import from_gps
from typing import Union
from sklearn.cluster import KMeans, DBSCAN

GPSParsable = Union[float, datetime, Time, str]
SupportedClusterAlgorithms = Union[KMeans, DBSCAN]

@dataclass
class ClusteredSeismicData:
    raw: TimeSeriesDict
    blrms: TimeSeriesDict
    clusters: SupportedClusterAlgorithms

    def vectors(self) -> np.ndarray:
        return np.array(
            list(self.blrms.values())
        ).T

    def times(self) -> Index:
        for ts in self.raw.values():
            if hasattr(ts, 'times'):
                return ts.times

        # all of the TimeSeries in self.raw should contain the times
        #   attribute, so this should never be hit
        assert False

    def n_clusters(self) -> int:
        if isinstance(self.clusters, KMeans):
            return self.clusters.n_clusters
        else:
            return len({label for label in self.clusters.labels_ if label >=0})

    def centers(
        self
    ) -> np.ndarray:
        '''
        calculate the centers of each cluster
    
        Returns
        -------
        np.ndarray
            the cluster centers, of shape (n_clusters, vector dimension)
        '''
        vectors = self.vectors()
        labels = self.clusters.labels_
        n_clusters = self.n_clusters()

        # organize vectors by cluster label.
        clusters = [np.empty(0) for _ in range(n_clusters)]
        for i in range(len(vectors)):
            if labels[i] not in range(n_clusters):
                # skip data that isn't in a cluster
                continue
    
            # put the vector in its proper cluster
            clusters[labels[i]] = np.append(clusters[labels[i]], [vectors[i]], axis=0)
    
        print(vectors)
        print(vectors.shape)
        for clust in clusters:
            print(clust)
            print(clust.shape)
            print(np.mean(clust, axis=0))

        return np.array([ np.mean(clust, axis=0) for clust in clusters ])


    def closest_index(
        self,
        vector: np.ndarray
    ) -> int:
        '''
        gets the index in self.vectors of the closest vector to the passed vector
    
        Parameters
        ----------
        vector : np.ndarray
            a vector of shape (1, number of TimeSeries in self.blrms)
    
        Returns
        -------
        int
            index of closest vector in self.vectors to the passed vector
        '''
        vectors = self.vectors()
        # assume to start that the closest index is the first vector in vectors
        closest_index = 0

        for i in range(len(vectors)):
            if np.linalg.norm(
                vector - vectors[i]
            ) < np.linalg.norm(
                vector - vectors[closest_index]
            ):
                closest_index = i

        return closest_index

    def psds_of_centers(
        self, 
        psd_length: float = 60,
    ) -> list[list[FrequencySeries]]:
        times = self.times()
        centers = self.centers()
        print(centers)
        print(centers.size)
        times_of_centers = [
            times[self.closest_index(center)].value
            for center in centers
        ]

        raw_cropped_to_centers = [
            self.raw.crop(
                start=times_of_centers[i],
                end=times_of_centers[i] + psd_length,
                copy=True
            )
            for i in range(len(centers))
        ]
        for i in range(len(centers)):
            print(centers[i], raw_cropped_to_centers[i].span)

        return [
            [cropped_ts.psd() for cropped_ts in raw_cropped_to_center.values()] 
            for raw_cropped_to_center in raw_cropped_to_centers
        ]
