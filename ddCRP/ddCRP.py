import numpy as np
from scipy import sparse

from ddCRP import sampling
from ddCRP import subgraphs
from ddCRP import statistics
from ddCRP import ward_clustering
import os
import csv

import time


class ddCRP(object):
    """
    Class to implement the distance-dependent Chinese Restaurant Process.

    Parameters:
    - - - - -
    alpha: float
        concentration parameter of CRP prior
    model: PriorBase
        type of prior model to use for feature data
    mcmc_passes: int
        number of MCMC passes to apply to data
        number of passes to run before recording statistics
    verbose: bool
        print statistics every stats_interval

    Notes:
    - - - -

    The '''model''' argument assumes that whatever object is provided is a
    subclass of the PriorBase.Prior abstract method.  The model must implement
    the followng three methods:

        - model.sufficient_statistics
        - model.posterior_parameters
        - model.marginal_evidence

    Any model that implements these methods will be accepted.  Currently, the
    only two models that are implemented are the Normal-Inverse-Chi-Squared
    and Normal-Inverse-Wishart models.  The NIX model treats each feature as
    independent from the others, while the NIW model does not make this
    assumption.
    """

    def __init__(
        self, alpha, model, mcmc_passes=100, stats_interval=500, parcelPath = "" , ward=False,
            n_clusters=7, verbose=True ):

        """
        Initialize ddCRP object.
        """

        self.alpha = np.float(alpha)
        self.model = model
        self.mcmc_passes = np.int(mcmc_passes)
        self.stats_interval = np.int(stats_interval)
        self.ward = ward
        self.n_clusters = n_clusters
        self.verbose = verbose
        self.parcelPath = parcelPath
        

    def fit(self, features, adj_list, init_z=None, init_c=None, gt_z=None, 
        edge_prior=None):

        """
        Main function to fit the distance-dependent Chinese Restaurant Process.
        Parameters:
        - - - - -
        features : array
                data array of features for each sample
        adj_list : dictionary
                adjacency list of samples
        init_z : array
            initialized cortical map
        init_c : array
                initialized parent linkage matrix
        gt_z : array
                ground truth map for computing normalized mutual information
        edge_prior : dictionary
                nested dictionary, probability of neighboring vertices beloning
                to same parcels
        """

        # initialize Gibbs sampling object
        gibbs = sampling.Gibbs()
        nvox = len(adj_list)

        # normalize each feature to have zero mean, unit variance
        features = statistics.Normalize(features)

        stats = {
            'times': [], 'lp': [], 'max_lp': [],
            'K': [], 'z': np.empty((0, nvox)),
            'c': np.empty((0, nvox)), 'NMI': [],
            'deltaC': [], 'boundC': []}

        # If initial clustering provided, generate spanning trees from it
        # Otherwise, if desired, generate spanning trees from Ward Clustering
        if np.any(init_z):
            init_c = subgraphs.ClusterSpanningTrees().fit(adj_list, init_z)
        elif self.ward:
            init_c = ward_clustering.Ward(features, adj_list, self.n_clusters)

        # compute initial MST component array matrix
        [c, G] = subgraphs.sparse_linkage(adj_list, nvox, init_c)

        # compute initial parcel count and parcel assignments
        [K, z, parcels] = subgraphs.connected_components(G)
        self.init_z = z

        # compute log-likelihood of initial cortical map
        curr_lp = self._fullProbabilityDDCRP(parcels, features)

        max_lp = -1.*np.inf
        map_z, boundC, deltaC = [], [], []
        t0 = time.time()
        steps = 0

        order = np.arange(nvox)

        # perform mcmc_passes of over all samples
        parcels_in_each_step = []
        for mcmc_pass in np.arange(self.mcmc_passes):

            # shuffle sample order for each MCMC pass
            np.random.shuffle(order)

            for i in order:

                # if current map log-probability greater than current max
                # set current map to best map
                if curr_lp > max_lp:
                    max_lp = curr_lp
                    map_z = z

                if steps % self.stats_interval == 0:
                    stats = statistics.UpdateStats(
                        stats, t0, curr_lp, max_lp,
                        K, list(z), list(c), steps,
                        gt_z, map_z, deltaC, boundC,
                        self.verbose)

                # remove current link to parent
                G[i, c[i]] = 0

                # if link was self-link
                if c[i] == i:
                    # Removing self-loop, parcellation won't change
                    rem_delta_lp = -np.log(self.alpha)
                    z_rem = z
                    parcels_rem = parcels
                else:
                    # otherwise compute new connected components
                    K_rem, z_rem, parcels_rem = subgraphs.connected_components(
                        G)

                    # if number of components changed
                    if K_rem != K:
                        # We split a cluster, compute change in likelihood
                        rem_delta_lp = -self._LogProbDifference(
                            parcels_rem, z_rem[i], z_rem[c[i]], features)

                    else:
                        rem_delta_lp = 0

                # get neighbors of sample i
                adj_list_i = adj_list[i]

                # initialize empty log-prob vector
                lp = np.zeros((len(adj_list_i)+1,))
                lp[-1] = np.log(self.alpha)

                for j, n in enumerate(adj_list_i):
                    # just undoing split
                    if z_rem[n] == z_rem[c[i]]:
                        lp[j] = -rem_delta_lp - (c[i] == i)*np.log(self.alpha)

                    # (possibly) new merge
                    elif z_rem[n] != z_rem[i]:
                        lp[j] = self._LogProbDifference(
                            parcels_rem, z_rem[i], z_rem[n], features)

                # sample new neighbor according to Gibbs
                new_neighbor = gibbs.sample(lp)
                if new_neighbor < len(adj_list_i):
                    c[i] = adj_list_i[new_neighbor]
                else:
                    c[i] = i

                # update current full log-likelihood with new parcels
                curr_lp = curr_lp + rem_delta_lp + lp[new_neighbor]
                # add new edge to parent graph
                G[i, c[i]] = 1
                # compute new connected components
                [K_new, z_new, parcels_new] = subgraphs.connected_components(G)

                deltaC = statistics.delta_C(parcels, parcels_new)
                boundC = statistics.boundaries(z_new, adj_list)
                K, z, parcels = K_new, z_new, parcels_new
                
                temp_parcel = [x+1 for x in z_new]
               
                parcels_in_each_step.append(temp_parcel)
                steps += 1

        # update diagnostic statistics
        stats = statistics.UpdateStats(
            stats, t0, curr_lp, max_lp, K,
            list(z), list(c), steps, gt_z,
            map_z, deltaC, boundC, self.verbose)

        # for visualization purposes
        map_z[np.where(map_z == 0)[0]] = map_z.max() + 1

        self.map_z_ = map_z
        self.stats_ = stats
        self._saveParcel(parcels_in_each_step)
    
    def _saveParcel(self , parcels_in_each_step):
        
      
        header = []
        if os.path.isfile(self.parcelPath):
          open(self.parcelPath, 'w').close()
        
        for i in range(len(parcels_in_each_step[0])):
            header.append("V" +str(i+1))
        
  
        f = csv.writer(open(self.parcelPath, 'a' , newline=''), delimiter=" ", quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #f.writerow(header)
        for i in range(len(parcels_in_each_step)): 
            f.writerow(parcels_in_each_step[i])
           
        
    def _fullProbabilityDDCRP(self, parcels, features):
        """
        Compute the full log-likelihood of the clustering.
        Parameters:
        - - - - -
        parcels : dictionary
                mapping between cluster ID and sample indices
        features : array
                data samples
        Returns:
        - - - -
        lp : float
                marginal log-likelihood of a whole parcelation
        """

        lp = self.model.full_evidence(parcels, features)

        return lp

    def _LogProbDifference(self, parcel_split, split_l1, split_l2, features):
        """
        Compute change in log-likelihood when considering a merge.

        Parameters:
        - - - - -
        parcel_split : dictionary
                mapping of cluster ID to sample indices
        split_l1 , split_l2 : int
                label values of components to merge
        features : array
                data samples
        Returns:
        - - - -
        ld : float
                log of likelihood ratio between merging and splitting
                two clusters
        """

        model = self.model

        merged_indices = np.concatenate([parcel_split[split_l1],
                                        parcel_split[split_l2]])

        # compute sufficient statistics, marginalized parameters
        # and log-likelihood of merged parcels
        merge_stats = model.sufficient_statistics(features[merged_indices, :])
        merge_phyp = model.posterior_parameters(merge_stats)
        merge_ll = model.marginal_evidence(merge_phyp, merge_stats)

        # compute likelihood of split parcels
        split_ll = self._LogProbSplit(
            parcel_split, split_l1, split_l2, features)

        ld = merge_ll - split_ll

        return ld

    def _LogProbSplit(self, parcel_split, split_l1, split_l2, features):
        """
        Compute change in log-likelihood when consiering a split.
        Parameters:
        - - - - -
        parcel_split : dictionary
                mapping of cluster ID to sample indices
        split_l1 , split_l2 : int
                label values of components to merge
        features : array
                data samples
        Returns:
        - - - -
        split_ll : float
                log-likelihood of two split clusters
        """

        model = self.model

        idx1 = parcel_split[split_l1]
        suff1 = model.sufficient_statistics(features[idx1, :])
        phyp1 = model.posterior_parameters(suff1)
        lp_1 = model.marginal_evidence(phyp1, suff1)

        idx2 = parcel_split[split_l2]
        suff2 = model.sufficient_statistics(features[idx2, :])
        phyp2 = model.posterior_parameters(suff2)
        lp_2 = model.marginal_evidence(phyp2, suff2)

        split_ll = lp_1 + lp_2

        return split_ll

