import rpy2.robjects as ro
import numpy as np

class PointClustering(object):
     
    def __init__(self, thinning = str(1), burnIn = str(1) , clust_trace_filepath = '' , method = 'avg' , max_k = str(4) ):
                 
        self.thinning = thinning
        self.burnIn = burnIn
        self.clust_trace_filepath = clust_trace_filepath
        self.method = method
        self.max_k = max_k

    def estimatePointClustering(self):
        
        ro.r('MCMCOptions=list(thinning='+ self.thinning+', burnIn= '+ self.burnIn+')') 
        ro.r('clust <- read.table(\''+ self.clust_trace_filepath +'\', stringsAsFactors=F)')
        ro.r('step <- MCMCOptions$thinning')
        ro.r('clust <- clust[seq(MCMCOptions$burnIn + 1, nrow(clust) , by=step), ]')
        ro.r(' if (any(is.na(clust))) { clust <- na.omit(clust)  }')

        ro.r('psm <- mcclust::comp.psm(as.matrix(clust))\n'+
             'tempMat <- diag(ncol(clust)) \n'+
             'if (all(psm == tempMat)){ mpear <- mcclust::maxpear(psm, method=\'avg\' , max.k = '+ str(self.max_k) +')\n'+
             'mpear$cl <- as.matrix(t(data.frame(best=mpear$cl, avg=mpear$cl, comp=mpear$cl, draws=mpear$cl)))} else {' +
             'mpear <- mcclust::maxpear(psm, method=\'all\', cls.draw = as.matrix(clust) , max.k = '+ str(self.max_k) +'  )}\n'+
             'cl <- mpear$cl')
        
        result = np.array(ro.r['cl'])
        if self.method == 'avg':
            return result[1]
        if self.method == 'comp':    
            return result[2]