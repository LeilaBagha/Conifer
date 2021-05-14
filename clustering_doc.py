# -*- coding: utf-8 -*-

from ddCRP import ddCRP
from ddCRP import Priors
from pointClustering import PointClustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



class DDCRP_OneDoc(object):

      def __init__(self, documents  , VAF , features ,mcmc_passes = 40 , stats_interval = 100 , max_k = 6, thinning = str(10) , burnIn = str(500) ,
                mu = 0, kappa = 0.01,nu = 70,sigma= 1 ,alpha = 10  ):
   
          self.mu = mu
          self.kappa =  kappa
          self.nu = nu
          self.sigma= sigma
          self.alpha = alpha
          
          self.mcmc_passes = mcmc_passes
          self.stats_interval = stats_interval
          
          self.max_k = max_k
          self.thinning = thinning
          self.burnIn = burnIn
          
          self.parcelPath = './clust-trace.csv'

          self.documents = documents
          self.VAF = VAF
          self.features = features
          self.dim = len(self.VAF)
          self.adj_list = {}.fromkeys(np.arange(self.dim ))
          
          
         
      def estimate_clutsering(self ):
          
          for i in range(len(self.features)):            
              
              curr_adj = []
              for d in range(len(self.documents)):
                  
                  if i in self.documents[d].words :
                      curr_adj.extend(self.documents[d].words)
              curr_adj = list(dict.fromkeys(curr_adj))       
              self.adj_list[i] = list(np.array(curr_adj)) 
          
          # dimensionality of data
          dimensionality_of_data = len(self.features[0])  
            
          np.random.seed(seed=2)
          mu_bar = np.zeros((dimensionality_of_data,))
          lambda_bar = np.random.rand(dimensionality_of_data,dimensionality_of_data) + np.eye(dimensionality_of_data)
          niw = Priors.NIW(mu0=mu_bar,kappa0=self.kappa,nu0=self.nu,lambda0=lambda_bar)
 
          crp = ddCRP.ddCRP(alpha = self.alpha, model=niw, mcmc_passes = self.mcmc_passes, stats_interval = self.stats_interval , parcelPath =self.parcelPath , ward=False ,  n_clusters=3 )
          crp.fit(self.features, self.adj_list)  
          
          pc = PointClustering(thinning = self.thinning, burnIn = self.burnIn , clust_trace_filepath = self.parcelPath , method = 'avg' , max_k = self.max_k )
          result_clustering = pc.estimatePointClustering()
          return result_clustering
          
         
         