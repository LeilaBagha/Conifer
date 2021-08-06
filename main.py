
from sc_data import InputData
from sampler import NCRPNode
from sampler import HierarchicalLDA
from clone_identifier import ddcrp_topic
from document import NCRPDocument 
import numpy as np


def conifer(corpus , vocab ,features,VAF, n_samples, seed, crp_alpha, gamma, eta, mcmc_passes, stats_interval, max_k, thinning, burnIn,
            mu, kappa, nu, sigma, ddcrp_alpha):
    
    hlda = HierarchicalLDA(corpus, vocab,features,VAF, n_samples, seed, crp_alpha, gamma, eta, mcmc_passes, stats_interval, max_k, thinning, burnIn, 
                           mu, kappa, nu, sigma, ddcrp_alpha)
    hlda.estimate()


def clone_identification(corpus, vocab, features, VAF, n_samples, mcmc_passes, stats_interval, max_k, thinning, burnIn, 
                         mu, kappa, nu, sigma, ddcrp_alpha):
                         
    features = np.array(list(features))
    
    documents = []
    
    for d in range(len(corpus)):   
        doc = NCRPDocument(doc_id = d, num_levels = max_k , root_node = None , vocab = vocab ,words = corpus[d])
        documents.append(doc)
        
    ddod = ddcrp_topic(documents , VAF, features , mcmc_passes = mcmc_passes , stats_interval = stats_interval , max_k = max_k, 
                       thinning = thinning , burnIn = burnIn, mu = mu, kappa = kappa,nu = nu, sigma= sigma , ddcrp_alpha = ddcrp_alpha)   
    result_clustering = ddod.estimate_clutsering()
    print(result_clustering)
    

  
if __name__ == '__main__':
    
    n_samples = 5
    
    #tree inference hyper-param
    crp_alpha=1 
    gamma=1.0 
    eta=1.0 
    num_levels=3
    seed = 0
    
    #clone identification hyper-param
    mcmc_passes = 150 
    stats_interval = 125 
    max_k = 3 
    thinning = str(1)  
    burnIn = str(1)
    mu = 0 
    kappa = 0.01
    nu = 70
    sigma= 1 
    ddcrp_alpha = 3
    
    
    data = InputData(bulk='./input/bulk.csv', singleCell ='./input/singleCell.csv', result= './output/', isVAF = 1)
    corpus , VAF , all_word_features, features = data.fetchData()
    
    # for tree inference
    conifer(corpus, data.vocab ,features,VAF, n_samples, seed, crp_alpha, gamma, eta, mcmc_passes, stats_interval, max_k, thinning, 
            burnIn, mu, kappa, nu, sigma, ddcrp_alpha)
    
    # just for clone identification
    clone_identification(corpus, data.vocab, all_word_features, VAF, n_samples, mcmc_passes, stats_interval, max_k, thinning, 
                        burnIn, mu, kappa, nu, sigma, ddcrp_alpha)
    
