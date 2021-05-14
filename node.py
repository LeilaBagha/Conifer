


from numpy.random import RandomState
import numpy as np

class NCRPNode(object):

    
    total_nodes = 0
    last_node_id = 0
    

    def __init__(self, num_levels, vocab, parent=None, level=0,
                 random_state=None):

        self.node_id = NCRPNode.last_node_id
        NCRPNode.last_node_id += 1

        self.customers = 0
        self.parent = parent
        self.children = []
        self.level = level
        self.total_words = 0
        self.num_levels = num_levels

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
        node = NCRPNode(self.num_levels, self.vocab, parent=self, level=self.level+1)
        self.children.append(node)
        NCRPNode.total_nodes += 1
        return node

    def is_leaf(self):
        
        return self.level == self.num_levels-1

    def get_new_leaf(self):
       
        node = self
        for l in range(self.level, self.num_levels-1):
            node = node.add_child()
        return node

    def drop_path(self):
        
        node = self
        node.customers -= 1
        if node.customers == 0:
            node.parent.remove(node)
        for level in range(1, self.num_levels): # skip the root
            node = node.parent
            node.customers -= 1
            if node.customers == 0:
                node.parent.remove(node)

    def remove(self, node):
        ''' Removes a child node '''
        self.children.remove(node)
        NCRPNode.total_nodes -= 1

    def add_path(self):
        
        node = self
        node.customers += 1
        for level in range(1, self.num_levels):
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
    
      
    def get_top_words(self, n_words, with_weight):
        

        pos = np.argsort(self.word_counts)[::-1]
        sorted_vocab = self.vocab[pos]
        sorted_vocab = sorted_vocab[:n_words]
        sorted_weights = self.word_counts[pos]
        sorted_weights = sorted_weights[:n_words]

        output = ''
        for word, weight in zip(sorted_vocab, sorted_weights):
            if weight != 0:
                if with_weight:
                    output += '%s (%d), ' % (word, weight)
                else:
                    output += '%s, ' % word
        return output