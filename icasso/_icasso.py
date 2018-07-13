""" Icasso based on Himberg et al. (2014)
"""
# Author: Erkka Heinila <erkka.heinila@jyu.fi>

import logging

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster

from sklearn.manifold import MDS


logger = logging.getLogger('icasso')


class Icasso(object):
    """ A container for Icasso
    """
    def __init__(self, ica_class, ica_params, iterations=10, 
                 bootstrap=True, vary_init=True):
        """
        """
        self._ica_class = ica_class
        self._ica_params = ica_params
        self._iterations = iterations
        self._bootstrap = bootstrap
        self._vary_init = vary_init

        self._store = []
        self._fit = False

    def fit(self, data, fit_params, random_state=None, bootstrap_fun=None, 
            unmixing_fun=None, store_fun=None):
        """ Use Icasso to given data.
        """
        if random_state is None:
            generator = np.random.RandomState()
        else:
            generator = np.random.RandomState(random_state)

        if self._vary_init:
            seeds = [generator.randint(0, 2**32) for i in 
                     range(self._iterations)]
        else:
            seeds = [generator.randint(0, 2**32)] * self._iterations

        components = []

        logger.info("Fitting ICA " + str(self._iterations) + " times.")
        for i in range(self._iterations):

            ica = self._ica_class(random_state=seeds[i], **self._ica_params)

            if self._bootstrap and not bootstrap_fun: 
                resampled_data = np.apply_along_axis(
                    func1d=generator.choice,
                    axis=1,
                    arr=data,
                    size=data.shape[1],
                    replace=True)
            elif self._bootstrap and bootstrap_fun:
                resampled_data = bootstrap_fun(data, generator)
            else:
                resampled_data = data

            ica.fit(resampled_data, **fit_params)

            if store_fun:
                self._store.append(store_fun(ica)) 
            
            if unmixing_fun:
                unmixing = unmixing_fun(ica)
            else:
                unmixing = ica.unmixing_

            components.extend(
                [component[0] for component in 
                 np.split(unmixing, unmixing.shape[0])])

        self._components = np.array(components)

        self._cluster(components)

        self._fit = True

    def _cluster(self, components):
        """ Apply agglomerative clustering with average-linkage criterion. """
        logger.info("Computing dissimilarity matrix")
        self._dissimilarity = np.sqrt(1 - np.abs(np.corrcoef(self._components)))
        self._linkage = linkage(squareform(self._dissimilarity, checks=False), 
                                method='average')

    def plot_dendrogram(self):
        """
        """
        if not self._fit:
            raise Exception("Model must be fitted before plotting") 

        logger.info("Plotting dendrogram..")
        plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        dendrogram(
            self._linkage,
            leaf_rotation=90.,
            leaf_font_size=1.,
            show_leaf_counts=False,
            no_labels=True,
        )
        plt.show()

    def plot_mds(self, distance=0.5, random_state=None):
        """
        """
        if not self._fit:
            raise Exception("Model must be fitted before plotting") 

        logger.info("Projecting components to 2D space with MDS")
        mds = MDS(n_components=2, max_iter=3000, eps=1e-9, 
                  random_state=random_state, dissimilarity="precomputed")

        mds.fit(self._dissimilarity)
        pos = mds.embedding_

        cluster_idxs = fcluster(self._linkage, distance, criterion='distance')

        logger.info("Plotting ICA components in 2D space..")
        colors = [np.random.rand(3,) for idx in range(max(cluster_idxs))]
        plt.figure()
        for idx in range(pos.shape[0]):
            plt.scatter(pos[idx,0], pos[idx,1], c=colors[cluster_idxs[idx]-1],
                        s=5)
        plt.show()

    def get_centroid_unmixing(self, distance=0.5):
        if not self._fit:
            raise Exception("Model must be fitted before plotting") 

        clusters_by_components = fcluster(self._linkage, distance, 
                                          criterion='distance')

        components_by_clusters = {}
        for comp_idx, cluster_id in enumerate(clusters_by_components):
            if cluster_id not in components_by_clusters:
                components_by_clusters[cluster_id] = []
            components_by_clusters[cluster_id].append(comp_idx)
        components_by_clusters = sorted(components_by_clusters.items(), 
                                        key=lambda x: x[0])
        components_by_clusters = [val for key, val in components_by_clusters]

        # Calculate quality index for compactness
        # and isolation
        similarities = np.abs(np.corrcoef(self._components))
        scores = []
        for idx, cluster in enumerate(components_by_clusters):
            other_clusters = components_by_clusters[:idx] + components_by_clusters[idx+1:]
            other_components = [comp for cluster in other_clusters for comp in cluster]
            within_sum = sum([similarities[ii, jj] for ii in cluster for jj in cluster])
            within_similarity = (1.0/len(cluster)**2)*within_sum
            between_sum = sum([similarities[ii, jj] for ii in cluster for jj in other_components]) 
            between_similarity = (1.0/(len(cluster)*len(other_components)))*between_sum
            scores.append(within_similarity - between_similarity)

        import pdb; pdb.set_trace()

        # get centrotype (maximum similarity to other points in the cluster)
        # order by compactness criterion
        # return centroids
        unmixing = None
        scores = None

        return unmixing, scores

