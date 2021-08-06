import csv
import os
from math import log
import sys

from node import NCRPNode

from ddCRP import ddCRP
from ddCRP import Priors
from pointClustering import PointClustering

from numpy.random import RandomState
from document import NCRPDocument
from clone_identifier import ddcrp_topic
import numpy as np


class HierarchicalLDA(object):

    def __init__(self, corpus, vocab,features, VAF, n_samples, seed, crp_alpha, gamma, eta, mcmc_passes, stats_interval, max_k, thinning, burnIn, 
                           mu, kappa, nu, sigma, ddcrp_alpha):

        NCRPNode.total_nodes = 0
        NCRPNode.last_node_id = 0
        
        self.num_samples = n_samples
        
        self.corpus = corpus
        self.vocab = vocab
        self.crp_alpha = crp_alpha  # smoothing on doc-topic distributions
        self.gamma = gamma          # "imaginary" customers at the next, as yet unused table
        self.eta = eta              # smoothing on topic-word distributions

        
        
        self.mcmc_passes = mcmc_passes
        self.stats_interval = stats_interval
        self.max_k = max_k
        self.thinning = thinning
        self.burnIn = burnIn
        self.VAF = VAF
                           
        self.mu = mu
        self.kappa = kappa
        self.nu = nu
        self.sigma = sigma
        self.ddcrp_alpha = ddcrp_alpha
        self.features = features
        
        self.clusterPath = './clust-trace.csv'
        self.seed = seed
        self.random_state = RandomState(seed)
        self.num_documents = len(corpus)
        self.num_types = len(vocab)
        self.eta_sum = eta * self.num_types
        self.root_node = NCRPNode(self.vocab)
        self.document_leaves = {}                                   # currently selected path (ie leaf node) through the NCRP tree
        self.levels = np.zeros(self.num_documents, dtype=np.object) # indexed < doc, token >
        self.documents = []
        
        for d in range(len(self.corpus)):   
            
           doc = NCRPDocument(doc_id = d, num_levels = 3 , root_node =  self.root_node ,
                              vocab = vocab ,words = self.corpus[d])
           self.documents.append(doc)    
        
        
        for d in range(len(self.documents)):
            document = self.documents[d]
            feature = self.features[d]
            level_assignment = self.clone_detection(document, feature) 
            document.num_levels = np.max(level_assignment)
        
        for d in range(len(self.documents)):
            
            
            # populate nodes into the path of this document
            doc = self.documents[d].words
            doc_num_levels = self.documents[d].num_levels+1
            path = np.zeros(doc_num_levels, dtype=np.object)
            doc_len = len(doc)
            path[0] = self.root_node
            self.root_node.customers += 1 # always add to the root node first
            
            for level in range(1, doc_num_levels):
                # at each level, a node is selected by its parent node based on the CRP prior
                parent_node = path[level-1]
                level_node = parent_node.select(self.gamma)
                level_node.level = level
                level_node.customers += 1
                path[level] = level_node

            # set the leaf node for this document
            leaf_node = path[doc_num_levels-1]
            
            self.document_leaves[d] = leaf_node

            # randomly assign each word in the document to a level (node) along the path
            self.levels[d] = np.zeros(doc_len, dtype=np.int)
            for n in range(doc_len):
                w = doc[n]
                random_level = self.random_state.randint(doc_num_levels)
                random_node = path[random_level]
                random_node.word_counts[w] += 1
                random_node.total_words += 1
                self.levels[d][n] = random_level

    def estimate(self):

        print ('HierarchicalLDA sampling\n')
        for s in range(self.num_samples):
            sys.stdout.write('.')
            for cd in range(len(self.documents)):
                self.sample_path(cd)

            for zd in range(len(self.documents)):
                self.sample_level(zd)

            #if (s > 0) and ((s+1) % display_topics == 0):
            print (" %d" % (s+1))
            self.print_nodes()
            print

    def sample_path(self, d):

        # define a path starting from the leaf node of this doc
        document = self.documents[d]
        doc_num_levels = document.num_levels+1
        path = np.zeros(doc_num_levels, dtype=np.object)
        node = self.document_leaves[d]
 
        for level in range(doc_num_levels-1, -1, -1): # e.g. [3, 2, 1, 0] for num_levels = 4
            path[level] = node
            node = node.parent

        # remove this document from the path, deleting empty nodes if necessary
        self.document_leaves[d].drop_path(doc_num_levels)

        ############################################################
        # calculates the prior p(c_d | c_{-d}) in eq. (4)
        ############################################################

        node_weights = {}
        self.calculate_ncrp_prior(node_weights, self.root_node, 0.0)

        ############################################################
        # calculates the likelihood p(w_d | c, w_{-d}, z) in eq. (4)
        ############################################################

        level_word_counts = {}
        for level in range(doc_num_levels):
            level_word_counts[level] = {}
        doc_levels = self.levels[d]
        words = document.words

        # remove doc from path
        for n in range(len(words)): # for each word in the doc

            # count the word at each level
            level = doc_levels[n]
            w = words[n]
            
            if w not in level_word_counts[level]:
                level_word_counts[level][w] = 1
            else:
                level_word_counts[level][w] += 1

            # remove word count from the node at that level
            level_node = path[level]
            level_node.word_counts[w] -= 1
            level_node.total_words -= 1
            assert level_node.word_counts[w] >= 0
            assert level_node.total_words >= 0

        self.calculate_doc_likelihood(node_weights, level_word_counts,doc_num_levels)

        ############################################################
        # pick a new path
        ############################################################

        nodes = np.array(list(node_weights.keys()))
        weights = np.array([node_weights[node] for node in nodes])
        weights = np.exp(weights - np.max(weights)) # normalise so the largest weight is 1
        weights = weights / np.sum(weights)

        choice = self.random_state.multinomial(1, weights).argmax()
        node = nodes[choice]

        # if we picked an internal node, we need to add a new path to the leaf
     
        if node.level > doc_num_levels-1:
          for i in range(doc_num_levels-1,node.level):
               temp_node = node.parent
               node = temp_node
             
        if node.level < doc_num_levels-1:   
          for l in range(node.level, doc_num_levels):
               node = node.add_child()
              
        # add the doc back to the path
        node.add_path(doc_num_levels)       # add a customer to the path
        self.document_leaves[d] = node      # store the leaf node for this doc

        # add the words
        for level in range(doc_num_levels-1, -1, -1): # e.g. [3, 2, 1, 0] for num_levels = 4
            word_counts = level_word_counts[level]
            for w in word_counts:
                node.word_counts[w] += word_counts[w]
                node.total_words += word_counts[w]
            node = node.parent

    
    def calculate_ncrp_prior(self, node_weights, node, weight):
        ''' Calculates the prior on the path according to the nested CRP '''

        for child in node.children:
            child_weight = log( float(child.customers) / (node.customers + self.gamma) )
            self.calculate_ncrp_prior(node_weights, child, weight + child_weight)

        node_weights[node] = weight + log( self.gamma / (node.customers + self.gamma))

    def calculate_doc_likelihood(self, node_weights, level_word_counts,doc_num_levels):

        # calculate the weight for a new path at a given level
        new_topic_weights = np.zeros(doc_num_levels)
        for level in range(1, doc_num_levels):  # skip the root

            word_counts = level_word_counts[level]
            total_tokens = 0

            for w in word_counts:
                count = word_counts[w]
                for i in range(count):  
                    new_topic_weights[level] += log((self.eta + i) / (self.eta_sum + total_tokens))
                    total_tokens += 1

        self.calculate_word_likelihood(node_weights, self.root_node, 0.0, level_word_counts, new_topic_weights, 0 , doc_num_levels)

    def calculate_word_likelihood(self, node_weights, node, weight, level_word_counts, new_topic_weights, level , doc_num_levels):

        # first calculate the likelihood of the words at this level, given this topic
        node_weight = 0.0
        word_counts = level_word_counts[level]
        total_words = 0

        for w in word_counts:
            count = word_counts[w]
            for i in range(count): 
                node_weight += log( (self.eta + node.word_counts[w] + i) /
                                    (self.eta_sum + node.total_words + total_words) )
                total_words += 1

        # propagate that weight to the child nodes
        for child in node.children :
            if level+1 < doc_num_levels:
                self.calculate_word_likelihood(node_weights, child, weight + node_weight,
                                           level_word_counts, new_topic_weights, level+1,doc_num_levels)

        # finally if this is an internal node, add the weight of a new path
        level += 1
        while level < doc_num_levels:
            node_weight += new_topic_weights[level]
            level += 1

        node_weights[node] += node_weight

    def sample_level(self, d):
         
        
       
        
        document = self.documents[d]
        feature = self.features[d]
        doc_num_levels = document.num_levels + 1
        
        
        
        level_assignment = self.clone_detection(document, feature) 
        #sort level assignment
        
        # initialise level counts
        doc_levels = self.levels[d]
        level_counts = np.zeros(doc_num_levels, dtype=np.int)
        for c in doc_levels:
            level_counts[c] += 1

        # get the leaf node and populate the path
        path = np.zeros(doc_num_levels, dtype=np.object)
        node = self.document_leaves[d]
        for level in range(doc_num_levels-1, -1, -1): # e.g. [3, 2, 1, 0] for num_levels = 4
            path[level] = node
            node = node.parent

        # put the word back into the model
        level_weights = np.zeros(doc_num_levels)
        words = document.words
        for n in range(len(words)):

            w = words[n]
            word_level = doc_levels[n]

            # remove from model
            level_counts[word_level] -= 1
            node = path[word_level]
            node.word_counts[w] -= 1
            node.total_words -= 1
            
            level = level_assignment[n]
            
            doc_levels[n] = level
            level_counts[level] += 1
            
            node = path[level]
            
            node.word_counts[w] += 1
            node.total_words += 1
    
    
    def clone_detection(self,document,feature):
          
          adj_list =   self.get_adj_list_by_co_occurence_frequency(document)
          feature = np.array(list(feature))
          
          dimensionality_of_data = len(self.VAF)
          np.random.seed(seed=2)
          mu_bar = np.zeros((dimensionality_of_data,))
          lambda_bar = np.random.rand(dimensionality_of_data, dimensionality_of_data) + np.eye(dimensionality_of_data)
          niw = Priors.NIW(mu0=mu_bar,kappa0=self.kappa,nu0=self.nu,lambda0=lambda_bar)
 
          crp = ddCRP.ddCRP(alpha = self.ddcrp_alpha, model=niw, mcmc_passes = self.mcmc_passes, stats_interval = self.stats_interval , parcelPath =self.clusterPath , ward=False ,  n_clusters=3 )
          crp.fit(feature, adj_list)  
          
          pc = PointClustering(thinning = self.thinning, burnIn = self.burnIn , clust_trace_filepath = self.clusterPath , method = 'avg' , max_k = self.max_k )
          result_clustering = pc.estimatePointClustering()
          
          #sort according to the VAF
          
          return result_clustering  
    
    def get_adj_list_by_co_occurence_frequency(self , document):
         
         adj_list = {}.fromkeys(np.arange(len(self.VAF)))
         this_doc_words = document.words
         
        
         words_documents = []
         for cnt in range(len(self.documents)):  
           words = self.documents[cnt].words
           row = np.zeros(len(self.vocab))
           for w in range(len(words)): 
               row[words[w]] = 1
           words_documents.append(row)
               
         words_frequencies = []
         for k in range(len(self.vocab)):
             words_frequencies.append(np.zeros(len(self.vocab)))
         
         counter = 0
         for cnt in range(len(words_documents)):
             row = np.zeros(len(self.vocab))
             for i in range(len(this_doc_words)):
                 for j in range(i+1 , len(this_doc_words)):
                     occur = words_documents[cnt][i] + words_documents[cnt][j]
                     if occur == 2:
                         counter +=1
                         words_frequencies[i][j] +=1
                         words_frequencies[j][i] +=1
         words_frequencies = [x / len(self.documents) for x in words_frequencies]
         
         mean_co_occurrence = sum(sum(words_frequencies))/counter
         
         curr_adj = []
         for i in range(len(this_doc_words)):
             curr_adj = []
             for j in range(len(this_doc_words)): 
                 if words_frequencies[i][j] > mean_co_occurrence :
                      curr_adj.append(j)
             curr_adj = list(dict.fromkeys(curr_adj))       
             adj_list[i] = list(np.array(curr_adj)) 
             
         return adj_list
            
                     
                        
    def print_nodes(self):
        temp = []
        self.print_node(self.root_node, 0,temp)

    def print_node(self, node, indent, temp):
        output = node.get_node_words(temp)
        out = '    ' * indent
        if indent !=0 and output !='':
            out += 'topic=%d level=%d (documents=%d): ' % (node.node_id, node.level, node.customers)
            out += output
            temp.extend(output.split(', '))
            print (out)
        for child in node.children:
            self.print_node(child, indent+1, temp)

def load_vocab(file_name):
    with open(file_name, 'rb') as f:
        vocab = []
        reader = csv.reader(f)
        for row in reader:
            idx, word = row
            stripped = word.strip()
            vocab.append(stripped)
        return vocab

def load_corpus(file_name):
    with open(file_name, 'rb') as f:
        corpus = []
        reader = csv.reader(f)
        for row in reader:
            doc = []
            for idx_and_word in row:
                stripped = idx_and_word.strip()
                tokens = stripped.split(' ')
                if len(tokens) == 2:
                    idx, word = tokens
                    doc.append(int(idx))
            corpus.append(doc)
        return corpus