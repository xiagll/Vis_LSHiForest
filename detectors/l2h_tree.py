#! /usr/bin/python
#
# Implemented by Xuyun Zhang (email: xuyun.zhang@auckland.ac.nz). Copyright reserved.
#

import numpy as np

from .l2h_node import *

class L2HashTree:
    def __init__(self, l2h_family=None, depth_limit=-1):
        self._l2h_family = l2h_family
        self._depth_limit = depth_limit
        self._root = None
        self._n_samples = 0 
        self._avg_branch_factor = 0
        self._reference_path_length = 0


    def fit(self, data):
        self._n_samples = len(data)
        self._depth_limit = self._depth_limit if self._depth_limit > 0 else np.inf if self._depth_limit == 0 else self.get_binary_random_height(self._n_samples)
        data = np.array(data)
        self._root = self._recursive_fit(data)
        self._avg_branch_factor = self._get_avg_branch_factor()
        self._reference_path_length = self.get_random_path_length_symmetric(self._n_samples)


    def _recursive_fit(self, data, pre_depth=-1):
        n_samples = len(data)
        
        if n_samples == 0:
            return None

        cur_depth = pre_depth + 1

        if len(np.unique(data, axis=0)) == 1 or cur_depth > self._depth_limit:
            return L2HashNode(len(data), None, {}, {})

        else:
            hash_function = self._l2h_family.get_hash_function()
            hash_function.fit(data)
            partition = self._split_data(data, hash_function)

            children_count = {}
            for key in partition.keys():
                children_count[key] = len(partition.get(key))

            children = {}
            for key in partition.keys():
                child_data = partition.get(key)
                children[key] = self._recursive_fit(child_data, cur_depth)
            
            return L2HashNode(len(data), hash_function, children, children_count)


    def _split_data(self, data, hash_function):
        ''' Split the data using the given hash function '''
        partition = {}
        for i in range(len(data)):
            key = hash_function.get_hash_value(np.array(data[i]))
            if key not in partition:
                partition[key] = [data[i]]
            else:
                sub_data = partition[key]
                sub_data.append(data[i])
                partition[key] = sub_data

        return partition


    def get_num_instances(self):
        return self._n_samples


    def display(self):
        self._recursive_display(self._root)	

    def _recursive_display(self, l2hash_node, leftStr=''):
        if l2hash_node is None:
            return

        print(leftStr+'('+str(len(leftStr))+'):'+str(l2hash_node))
        
        children = l2hash_node.get_children()

        for key in children.keys():
            self._recursive_display(children[key], leftStr+' ')


    def decision_function(self, x):
        path_length = self._recursive_get_search_depth(self._root, 0, x)
        return pow(2.0, -1.0*path_length/self._reference_path_length)


    def _recursive_get_search_depth(self, l2hash_node, cur_depth, x):
        if l2hash_node is None:
            return -1

        children = l2hash_node.get_children()
        if not children:
            adjust_factor = self.get_random_path_length_symmetric(l2hash_node.get_data_size())
            return cur_depth+adjust_factor
        else:
            key = l2hash_node.get_hash_function().get_hash_value(x)
            if key in children.keys():
                return self._recursive_get_search_depth(children[key], cur_depth+1, x)
            else:
                return cur_depth+1


    def get_avg_branch_factor(self):
        return self._avg_branch_factor


    def _get_avg_branch_factor(self):
        i_count, bf_count = self._recursive_sum_BF(self._root)
        
        # Single node PATRICIA trie
        if i_count == 0:
            return 2.0

        return bf_count*1.0/i_count	


    def _recursive_sum_BF(self, l2hash_node):
        if l2hash_node is None:
            return None, None

        children = l2hash_node.get_children()
        if not children:
            return 0, 0
        else:
            i_count, bf_count = 1, len(children)
            for key in children.keys():
                i_c, bf_c = self._recursive_sum_BF(children[key])
                i_count += i_c
                bf_count += bf_c
            return i_count, bf_count


    def get_random_path_length_symmetric(self, num_samples):
        if num_samples <= 1:
            return 0
        elif num_samples > 1 and num_samples <= round(self._avg_branch_factor):
            return 1
        else:
            return (np.log(num_samples)+np.log(self._avg_branch_factor-1.0)+0.5772)/np.log(self._avg_branch_factor)-0.5


        # Binary tree has the highest height
    def get_binary_random_height(self, num_samples):
        return 2*np.log2(num_samples)+0.8327
