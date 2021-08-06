import numpy as np
from numpy.random import RandomState

class NCRPNode(object):
    
    total_nodes = 0
    last_node_id = 0

    def __init__(self, vocab, parent=None, level=0,
                 random_state=None):

        self.node_id = NCRPNode.last_node_id
        NCRPNode.last_node_id += 1

        self.customers = 0
        self.parent = parent
        self.children = []
        self.level = level
        self.total_words = 0
        self.vocab = np.array(vocab)
        self.word_counts = np.zeros(len(vocab))

        if random_state is None:
            self.random_state = RandomState()
        else:
            self.random_state = random_state

    def __repr__(self):
        parent_id = None
        if self.parent is not None:
            parent_id = self.parent.node_id
        return 'Node=%d level=%d customers=%d total_words=%d parent=%s' % (self.node_id,
            self.level, self.customers, self.total_words, parent_id)

    def add_child(self):
        ''' Adds a child to the next level of this node '''
        node = NCRPNode(self.vocab, parent=self, level=self.level+1)
        self.children.append(node)
        NCRPNode.total_nodes += 1
        return node

    def is_leaf(self,num_levels):
        ''' Check if this node is a leaf node '''
        return self.level == num_levels-1

    def get_new_leaf(self,num_levels):
        ''' Keeps adding nodes along the path until a leaf node is generated'''
        node = self
        for l in range(self.level, num_levels):
            node = node.add_child()
        return node

    def drop_path(self,num_levels):
        ''' Removes a document from a path starting from this node '''
        node = self
        node.customers -= 1
        if node.customers == 0:
            node.parent.remove(node)
        for level in range(1, num_levels): # skip the root
            node = node.parent
            node.customers -= 1
            if node.customers == 0:
                node.parent.remove(node)

    def remove(self, node):
        self.children.remove(node)
        NCRPNode.total_nodes -= 1

    def add_path(self,num_levels):
        node = self
        node.customers += 1
        for level in range(1, num_levels):
            node = node.parent
            node.customers += 1

    def select(self, gamma):
        weights = np.zeros(len(self.children)+1)
        weights[0] = float(gamma) / (gamma+self.customers)
        i = 1
        
        for child in self.children:
            weights[i] = float(child.customers) / (gamma + self.customers)
            i += 1
            
        choice = self.random_state.multinomial(1, weights).argmax()
       
        if choice == 0:
            return self.add_child()
        else:
            return self.children[choice-1]

    
    def get_node_words(self, temp):
        
        output = ''
        for i in range(len(self.word_counts)):
            if self.word_counts[i] != 0:
                if self.vocab[i] not in temp:
                    output += '%s, ' % self.vocab[i]
        return output        
