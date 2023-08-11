#! /usr/bin/python
#
# Implemented by Xuyun Zhang (email: xuyun.zhang@auckland.ac.nz). Copyright reserved.
#

import numpy as np

from .l2h_tree import L2HashTree

class L2HashForest:
        def __init__(self, num_trees, sampler, l2h_family):
                self._num_trees = num_trees
                self._sampler = sampler
                self._l2h_family = l2h_family
                self._trees = []

        
        def display(self):
                for t in self._trees:
                        t.display()


        def fit(self, data):
                # Important: clean the tree array
                self._trees = []

                # Use the first row to index the data, mainly for sampling
                indices = range(len(data))
                indexed_data = np.c_[indices, data]
                # Sampling data
                self._sampler.fit(indexed_data)
                sampled_indexed_datas = self._sampler.draw_samples(indexed_data)
                # Transform back the data
                sampled_datas = []
                for sampled_indexed_data in sampled_indexed_datas:
                        sampled_datas.append(np.array(sampled_indexed_data)[:, 1:])
                
                # Build Learning to Hash (L2Hash) trees
                for i in range(self._num_trees):
                        sampled_data = sampled_datas[i]
                        tree = L2HashTree(self._l2h_family)
                        tree.fit(sampled_data)
                        self._trees.append(tree)

        
        def decision_function(self, data):
                scores=[]
                data_size = len(data)
                for i in range(data_size):
                        d_scores = []
                        for j in range(self._num_trees):
                                transformed_data = data[i]
                                if self._sampler._bagging != None:
                                        transformed_data = self._sampler._bagging_instances[j].get_transformed_data(np.mat(data[i])).A1
                                d_scores.append(self._trees[j].decision_function(transformed_data))
                        scores.append(d_scores)
        
                avg_scores=[]
                for i in range(data_size):
                        score_avg = 0.0
                        for j in range(self._num_trees):
                                score_avg += scores[i][j]
                        score_avg /= self._num_trees
                        avg_scores.append(score_avg)

                return -1.0*np.array(avg_scores)


        def get_avg_branch_factor(self):
                sum = 0.0
                for t in self._trees:
                        sum += t.get_avg_branch_factor()
                return sum/self._num_trees              
