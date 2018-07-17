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

from scipy.spatial import ConvexHull

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

        self.store = []
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
                sample_idxs = generator.choice(range(data.shape[1]), size=data.shape[1])
                resampled_data = data[:, sample_idxs]
            elif self._bootstrap and bootstrap_fun:
                resampled_data = bootstrap_fun(data, generator)
            else:
                resampled_data = data

            ica.fit(resampled_data, **fit_params)

            if store_fun:
                self.store.append(store_fun(ica)) 
            
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
        """ Apply agglomerative clustering with average-linkage criterion and 
        correlation-based dissimilarity. """
        self._dissimilarity = np.sqrt(1 - np.abs(np.corrcoef(self._components)))
        self._linkage = linkage(squareform(self._dissimilarity, checks=False), 
                                method='average')

    def plot_dendrogram(self):
        """ Plots dendrogram of the linkage
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

    def plot_mds(self, distance=0.8, random_state=None):
        """ Plots components projected to 2d space and draws hulls around clusters
        """
        if not self._fit:
            raise Exception("Model must be fitted before plotting") 

        logger.info("Projecting components to 2D space with MDS")
        mds = MDS(n_components=2, max_iter=3000, eps=1e-9, 
                  random_state=random_state, dissimilarity="precomputed")

        mds.fit(self._dissimilarity)
        pos = mds.embedding_

        clusters_by_components = fcluster(self._linkage, distance, criterion='distance')

        components_by_clusters = self._get_components_by_clusters(
            clusters_by_components)

        # compute hulls for clusters
        convex_hulls = [
            pos[np.array(cluster)[ConvexHull([pos[idx] for idx 
                                              in cluster]).vertices]]
                        for cluster in components_by_clusters]

        scores = self._get_scores(components_by_clusters)

        logger.info("Plotting ICA components in 2D space..")

        # plot components as points in 2d plane
        plt.figure()
        sc = plt.scatter(pos[:, 0], pos[:, 1], c='red', s=5)

        # draw hulls
        sorted_hulls = [hull for hull, _ in sorted(zip(convex_hulls, scores),
                                                   key=lambda x: x[1],
                                                   reverse=True)]
        for hull_idx, hull in enumerate(sorted_hulls):
            leftmost_pos = None
            for idx in range(len(hull)):
                start_idx, end_idx = idx, idx+1
                if end_idx == len(hull):
                    end_idx = 0

                x = hull[start_idx][0], hull[end_idx][0]
                y = hull[start_idx][1], hull[end_idx][1]
                plt.plot(x, y, color='black')

                if leftmost_pos is None or hull[start_idx][0] < leftmost_pos[0]:
                    leftmost_pos = hull[start_idx]
            xlim = plt.xlim()
            plt.text(leftmost_pos[0] - 0.03*(xlim[1] - xlim[0]), leftmost_pos[1], str(hull_idx+1))
            
        plt.show()

    def get_centrotype_unmixing(self, distance=0.8):
        if not self._fit:
            raise Exception("Model must be fitted before plotting") 

        clusters_by_components = fcluster(self._linkage, distance, 
                                          criterion='distance')

        components_by_clusters = self._get_components_by_clusters(
            clusters_by_components)
        scores = self._get_scores(components_by_clusters)
        similarities = np.abs(np.corrcoef(self._components))

        centrotypes = []
        for cluster in components_by_clusters:
            centrotype = None
            max_simsum = None
            for component_idx in cluster:
                simsum = sum([similarities[component_idx, other_idx] for 
                               other_idx in cluster])
                if max_simsum == None or simsum > max_simsum:
                    max_simsum = simsum
                    centrotype = self._components[component_idx]

            centrotypes.append(centrotype)

        unmixing_matrix = np.array([centrotype for centrotype, _ in 
                                    sorted(zip(centrotypes, scores), 
                                           key=lambda x: x[1])])

        return unmixing_matrix[::-1, :], sorted(scores)[::-1]

    def _get_components_by_clusters(self, clusters_by_components):
        """ Converts clusters-by-components representation to
        components-by-clusters """
        components_by_clusters = {}
        for comp_idx, cluster_id in enumerate(clusters_by_components):
            if cluster_id not in components_by_clusters:
                components_by_clusters[cluster_id] = []
            components_by_clusters[cluster_id].append(comp_idx)
        components_by_clusters = sorted(components_by_clusters.items(), 
                                        key=lambda x: x[0])
        components_by_clusters = [val for key, val in components_by_clusters]
        return components_by_clusters
        

    def _get_scores(self, components_by_clusters):
        """ Calculate quality index for compactness
        and isolation """
        similarities = np.abs(np.corrcoef(self._components))
        scores = []
        
        for idx, cluster in enumerate(components_by_clusters):
            other_clusters = components_by_clusters[:idx] + components_by_clusters[idx+1:]
            other_components = [comp for cluster_ in other_clusters for comp in cluster_]
            within_sum = sum([similarities[ii, jj] for ii in cluster for jj in cluster])
            within_similarity = (1.0/len(cluster)**2)*within_sum
            between_sum = sum([similarities[ii, jj] for ii in cluster for jj in other_components]) 
            between_similarity = (1.0/(len(cluster)*len(other_components)))*between_sum
            scores.append(within_similarity - between_similarity)

        return scores
