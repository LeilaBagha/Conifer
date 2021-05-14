# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:43:59 2020

@author: shamisa
"""
import numpy as np

class NCRPDocument(object):
    
     def __init__(self, doc_id , num_levels, root_node, vocab ,words, leaf =None ):
         
         self.doc_id = doc_id
         self.num_levels = num_levels
         self.vocab = vocab
         self.words = words
         self.root_node = root_node
         self.doc_len = len(words)
         self.leaf = leaf
         
         self.path = np.zeros(self.num_levels, dtype=np.object)
         self.path[0] = self.root_node
         self.levels = np.zeros(self.doc_len, dtype=np.int)
     
     def __repr__(self):
        return 'Documnet=%d ' % (self.doc_id)   
    
     def reset_leaf(self,leaf,number_of_levels):
        self.leaf = leaf
        node = leaf
        for level in range(number_of_levels-1, -1, -1): # e.g. [3, 2, 1, 0]
            self.path[level] = node
            node = node.parent
            
