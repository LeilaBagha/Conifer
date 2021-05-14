# -*- coding: utf-8 -*-

import csv
from math import log
import sys
from numpy.random import RandomState
import numpy as np
from node import NCRPNode
from document import NCRPDocument
from clustering_OneDoc import DDCRP_OneDoc
        
from ddCRP import ddCRP
from ddCRP import Priors


class HierarchicalLDA(object):

    def __init__(self, corpus, vocab,
                 alpha=10.0, gamma=1.0, eta=0.1,
                 seed=0, verbose=True, num_levels=3 , VAF = [] , fileName = '' , resultPath = ''):

        NCRPNode.total_nodes = 0
        NCRPNode.last_node_id = 0

        self.corpus = corpus
        self.vocab = vocab
        self.alpha = alpha  
        self.gamma = gamma  
        self.eta = eta      
        self.fileName = fileName
        self.resultPath = resultPath
        self.VAF = VAF

        self.seed = seed
        self.random_state = RandomState(seed)
        self.verbose = verbose

        self.num_levels = num_levels
        self.num_documents = len(corpus)
        self.num_types = len(vocab)
        self.eta_sum = eta * self.num_types
        self.documents = []
        self.total_nodes = 0 
   
        self.root_node = NCRPNode(self.num_levels, self.vocab)

        
             
        for d in range(len(self.corpus)):

            
            
            doc = NCRPDocument(d, self.num_levels,self.root_node, self.vocab ,self.corpus[d])
            self.documents.append(doc)
            self.root_node.customers += 1 
           
            for level in range(1, self.num_levels):
                
                parent_node = doc.path[level-1]
                level_node = parent_node.select(self.gamma)
                level_node.customers += 1
                doc.path[level] = level_node

            
            doc.leaf = doc.path[self.num_levels-1]
 
            
            for n in range(doc.doc_len):
                w = doc.words[n]
                random_level = self.random_state.randint(self.num_levels)
                
                random_node = doc.path[random_level]
                random_node.word_counts[w] += 1
                random_node.total_words += 1
                
                doc.levels[n] = random_level
            
       
    def estimate(self, num_samples, display_topics=50, n_words=5, with_weights=True):

        print ('HierarchicalLDA sampling\n')
        for s in range(num_samples):
          
            print("================== sample %d =====================" % (s))
            
            sys.stdout.write('.')
            
            for d in range(len(self.corpus)):
                self.sample_path(d)
                #self.sample_level(d)
           
            for d in range(len(self.corpus)): 
                self.sample_level(d)
               
            print ("Sample: %d" % (s+1))
            self.print_nodes(n_words, with_weights ,self.fileName.split("/")[len(self.fileName.split("/"))-1].split(".csv")[0]+'.tree' )
            print
        self.save_clustering_result(self.root_node, n_words , self.fileName.split("/")[len(self.fileName.split("/"))-1].split(".csv")[0]+'.clusters') 

         
    def sample_path(self, d):

        doc = self.documents[d]
       
        node = doc.leaf
        number_of_levels = node.level+1
    
   
        path = np.zeros(number_of_levels, dtype=np.object)
        
        for level in range(number_of_levels-1, -1, -1): 
            path[level] = node
            node = node.parent

       
        doc.leaf.drop_path()
        doc.path = np.zeros(self.num_levels, dtype=np.object)

        node_weights = {}
        self.calculate_ncrp_prior(node_weights, self.root_node, 0.0)

        level_word_counts = {}
        for level in range(number_of_levels):
            level_word_counts[level] = {}
       

   
        for n in range(len(doc.words)):
            
       
            level = doc.levels[n] 
            w = doc.words[n]
           
            if w not in level_word_counts[level]:
                level_word_counts[level][w] = 1
            else:
                level_word_counts[level][w] += 1
                
           
            path[level].word_counts[w] -= 1
            path[level].total_words -= 1
                 
            assert path[level].word_counts[w] >= 0
            assert path[level].total_words >= 0

        self.calculate_doc_likelihood(node_weights, level_word_counts, number_of_levels)

       
        nodes = np.array(list(node_weights.keys()))
        weights = np.array([node_weights[node] for node in nodes])
        weights = np.exp(weights - np.max(weights)) 
        weights = weights / np.sum(weights)

        choice = self.random_state.multinomial(1, weights).argmax()
        
        node = nodes[choice]

   
        if not node.is_leaf():
            node = node.get_new_leaf()

      
        node.add_path()                     
        doc.reset_leaf(node, node.level+1)
        
        # add the words
        for level in range(doc.num_levels-1, -1, -1): 
            word_counts = level_word_counts[level]
            for w in word_counts:
                node.word_counts[w] += word_counts[w]
                node.total_words += word_counts[w]
            node = node.parent
        
        
    
    def calculate_ncrp_prior(self, node_weights, node, weight):
      
        
        for child in node.children:
            child_weight = log( float(child.customers) / (node.customers + self.gamma) )
            self.calculate_ncrp_prior(node_weights, child, weight + child_weight)

        node_weights[node] = weight + log( self.gamma / (node.customers + self.gamma))

    def calculate_doc_likelihood(self, node_weights, level_word_counts,number_of_levels):

       
        new_topic_weights = np.zeros(number_of_levels)
        for level in range(1, number_of_levels):  

            word_counts = level_word_counts[level]
            total_tokens = 0

            for w in word_counts:
                count = word_counts[w]
                for i in range(count):  
                    new_topic_weights[level] += log((self.eta + i) / (self.eta_sum + total_tokens))
                    total_tokens += 1

        self.calculate_word_likelihood(node_weights, self.root_node, 0.0, level_word_counts, new_topic_weights, 0,number_of_levels)

    def calculate_word_likelihood(self, node_weights, node, weight, level_word_counts, new_topic_weights, level,number_of_levels):

        
        node_weight = 0.0
        word_counts = level_word_counts[level]
        total_words = 0

        for w in word_counts:
            count = word_counts[w]
            for i in range(count): 
                node_weight += log( (self.eta + node.word_counts[w] + i) /
                                    (self.eta_sum + node.total_words + total_words) )
                total_words += 1

      
        for child in node.children:
            if child.level < number_of_levels:
                self.calculate_word_likelihood(node_weights, child, weight + node_weight,
                                           level_word_counts, new_topic_weights, level+1 , number_of_levels)

     
        level += 1
        while level < number_of_levels:
            node_weight += new_topic_weights[level]
            level += 1
        node_weights[node] += node_weight
     
    def sample_level(self, d):

      
        doc = self.documents[d]
  
        documets_with_same_path_vocab_list = [] 
        path_documents = []
          
        for i in range(len(self.documents)):
            documets_with_same_path_vocab_list.append(np.zeros(len(self.vocab), dtype=np.float))
        

        for doc_node in doc.path: 
            for i in range(len(self.documents)):
               od = self.documents[i]
               for od_node in od.path: 
                   if od_node.node_id == doc_node.node_id : 
                       path_documents.append(od)
                       vocabs = documets_with_same_path_vocab_list[i]
                       for l in range(len(od.levels)):   
                          if od.levels[l] == doc_node.level:
                             vocab_idx = self.vocab.index('X'+str(od.words[l]+1))
                             vocabs[vocab_idx] = 1
                          
                       documets_with_same_path_vocab_list[i] = vocabs      
        
        words_document = list(map(list, zip(*documets_with_same_path_vocab_list))) 
        words_document = np.array(words_document)
        
        freq_levels = np.zeros(len(self.vocab), dtype=np.object) 
        
        for j in range(len(documets_with_same_path_vocab_list[0])):
            
            word_freq = np.zeros(len(list(dict.fromkeys(doc.levels))), dtype=np.float)
            
            for i in range(len(documets_with_same_path_vocab_list)):
                if (documets_with_same_path_vocab_list[i])[j] == 1 and ((path_documents[i]).levels[(path_documents[i]).words.index(j)]) < len(word_freq) :
                    word_freq[(path_documents[i]).levels[(path_documents[i]).words.index(j)]] +=1
            freq_levels[j] = word_freq      
            
       
        total_word_freq = np.zeros(len(list(dict.fromkeys(doc.levels))), dtype=np.float)
        for k in range(len(freq_levels)):
            for l in range(len(total_word_freq)):
              total_word_freq[l] += (freq_levels[k])[l]
        
        features = np.zeros(len(self.vocab), dtype=np.object)
        
        for k in range(len(freq_levels)):
            features[k] = np.zeros(len(list(dict.fromkeys(doc.levels)))+1, dtype=np.float)
            (features[k])[0] = self.VAF[k]
            for  l in range(len(total_word_freq)): 
             (features[k])[l+1] = float(((freq_levels[k])[l])/total_word_freq[l])
         
        features = np.array(list(features))
        ddod = DDCRP_OneDoc(self.documents , self.VAF , features)   
        result_clustering = ddod.estimate_clutsering()

                       
    def print_nodes(self, n_words, with_weights,filename):
        self.print_node(self.root_node, 0, n_words, with_weights ,filename)

    def print_node(self, node, indent, n_words, with_weights,filename):
        
        nodes_each_level = node.get_top_words(n_words, with_weights)
        
        out = '    ' * indent
        out += 'topic=%d level=%d (documents=%d): ' % (node.node_id, node.level, node.customers)
        out += nodes_each_level
        
        path = self.resultPath + filename
        f = open(path, "a")
        f.write(out+"\n")
        f.close()

        
        print (out)
        for child in node.children:
            self.print_node(child, indent+1, n_words, with_weights,filename)
    
    
    
    def save_clustering_result(self, node, n_words , filename):
        
        nodes_each_level = node.get_top_words(n_words, False)
        path = self.resultPath + filename
        f = open(path, "a")
        text = '%d, ' % (node.node_id)
        text += nodes_each_level +'\n'
        f.write(text)
        f.close()
        
        for child in node.children:
            self.save_clustering_result(child,  n_words, filename)
            
    def get_total_nodes(self,node):
        
        if(node.level == 0):
            self.total_nodes = 1
        if len(node.children) != 0 :    
            self.total_nodes += len(node.children)
            for i in node.children : 
                self.get_total_nodes(i)
        return self.total_nodes          

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
    
               