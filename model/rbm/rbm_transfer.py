import tensorflow as tf
import copy
import sys
import numpy as np
import os
import pickle
import itertools

class RBMTransfer(object):
    """
    handle the transfer for multilayer perceptron model
    """

    def __init__ (self, model_target, graph_target, base_model_path, base_model_number=None, divide=1.):
        """
        Initialize an RBM transfer object.
        Args:
            model_target: the target model that we want to transfer the params
            graph_target: the graph of the target task
            base_model_path: the path of the base model that will be transferred
            base_model_number: the number of the base model that wants to be transffered (number of experiments, default: None, we get the latest model)
            divide: to divide the parameters by some values to avoid nan
        """
        self.model_target = model_target
        self.graph_target = graph_target
        self.base_model_path = base_model_path
        self.base_model_number = base_model_number 
        self.divide = divide
        self.initialize()


    def initialize(self):
        """
        Initialize the transfer
        """
        # Get the base model from the path
        self.learner_base = self.get_base_model()
        self.model_base = self.learner_base.model
        self.graph_base = self.learner_base.hamiltonian.graph

        # Initialize the transferred weight and biases from the target model
        self.W_transfer = self.model_target.W_array 
        self.bv_transfer = self.model_target.bv_array 
        self.bh_transfer = self.model_target.bh_array 

        self.params_base = self.model_base.get_parameters()

        # Divide parameter of the base network
        self.W_base = self.params_base[0] / self.divide
        self.bv_base = self.params_base[1] / self.divide
        self.bh_base = self.params_base[2] / self.divide


    def get_base_model(self):
        """
        Load the base learner model.
        Return:
            base learner model
            
        """
        ## If base model number is not specified, get the latest trained model
        if self.base_model_number is None:
            dir_names = [int(f) for f in os.listdir(self.base_model_path) if os.path.isdir(self.base_model_path + f)]
            self.base_model_number = max(dir_names)

        ## Load the base model
        self.base_model_path = '%s/%d/model.p' % (self.base_model_path, self.base_model_number)
        base_model = pickle.load(open(self.base_model_path, 'rb'))
        return base_model


    def cutpaste(self):
        """
        If the base model and target model is the same, we just cut and paste the parameters for the transfer.
        """
        assert self.model_target.num_visible >= self.model_base.num_visible, "Number of visible node in the model must be larger than or equal to the numbero f visible node in the base model!"
        self.model_target.set_parameters(self.params_base) 

    def expand_hidden(self, k_val):
        """
        Transfer for expanding hidden layer.
        """

        half_col = self.W_transfer.shape[1] // 2
        multiplier = self.W_transfer.shape[1] // self.W_base.shape[1]

        self.bv_transfer = self.bv_base

        if k_val == 0:
            self.W_transfer[:,:half_col] = self.W_base
            self.bh_transfer[:,:half_col] = self.bh_base
        elif k_val == 1:
            self.W_transfer = np.repeat(self.W_base, multiplier, 1)
            self.bh_transfer = np.repeat(self.bh_base, multiplier, 1)
        else:
            self.W_transfer = np.tile(self.W_base, (1, multiplier))
            self.bh_transfer = np.tile(self.bh_base, (1, multiplier))


        self.model_target.set_parameters([self.W_transfer, self.bv_transfer, self.bh_transfer])



    def tiling (self, k_val):
        assert self.model_target.num_visible >= self.model_base.num_visible and self.model_target.num_visible % self.model_base.num_visible == 0, "Number of visible node in the model must be larger than or equal to and divisible by the number of visible node in the base model!"
        assert self.graph_base.length % k_val == 0, "k must be divisible by the number of visible node in base model!"
        
        p_val = int(self.graph_target.length / self.graph_base.length)


        base_coor = []
        for point in range(self.graph_base.num_points):
            ##### Map old coordinate to the new coordinate which is the old_coor * the k_size
            ## For instance:
            ## 1D from 4 to 8 particles
            ## o-o-o-o  to o-o-o-o-o-o-o-o 
            ## 0 1 2 3  to 0 1 2 3 4 5 6 7
            ## 0 will be transferred to 0
            ## 1 will be transferred to 2 and so on
            ##
            ## 2D from 2x2 to 4x4
            ## 0,0    0,1        0,0     0,1    0,2    0,3
            ##  o------o          o-------o------o------o
            ##  |      |          |      1|1    1|2     |
            ##  o------o     1,0  o-------o------o------o 1,3
            ## 1,0    1,1         |      2|1    2|2     |
            ##               2,0  o-------o------o------o 2,3
            ##                    |      3|1    3|2     |
            ##               3,0  o-------o------o------o 3,3
            ## 0,0 will be transfered to 0,0
            ## 1,0 will be tranferred to 2,0
            ## and so on. 
            ## Similar for 3D
            old_coor = np.array(self.graph_base._point_to_coordinate(point))
                
            ## map the first position of the old coordinate in the base network to the new coordinate in the target network
            new_coor = (old_coor // k_val) * (k_val * p_val) + (old_coor % k_val)
    
            ##### Generate all possible combinations for the product 
            ## We want to transfer 0 to 0 and 1 for 1D
            ## and 0,0 to 0,0; 0,1; 1,0; 1,1 for 2D 
            ## We generate all possible combinations for the product
            ## For instance: 
            ## 1D from 4 to 8 particles
            ## old_coor 0 -> new_coor 0 -> to_iter = [[0,1]]
            ## old_coor 2 -> new coor 4 -> to_iter = [[4,5]]
            ## 1D from 4 to 16 particles
            ## old_coor 0 -> new_coor 0 -> to_iter = [[0, 1, 2, 4]
            ## old_coor 2 -> new_coor 8 -> to_iter = [[8, 9, 10, 11]]
            ## 2D from 2x2 to 4x4 particles
            ## old_coor 0,0 -> new_coor 0,0 -> to_iter = [[0,1],[0,1]]
            ## old_coor 1,0 -> new_coor 2,0 -> to_iter = [[2,3],[0,1]]
            ## 3D from 2x2x2 to 4x4x4
            ## old_coor (0,0,0) -> new_coor 0,0,0 -> to_iter=[[0,1], [0,1], [0,1]]
            ## old_coor (1,0,1) -> new_coor 2,0,2 -> to_iter=[[2,3], [0,1], [2,3]]
            ##
            ## because later we will do a product multiply on the to_iter to generate all possible combinations except for 1D
            ## 2D from 2x2 to 4x4 
            ## new_coor 0,0 -> to_iter = [[0,1],[0,1]] do a product multiply
            ## [0,1] x [0,1] = [[0,0], [0,1], [1,0], [1,1]
            ## new_coor 2,0 -> to_iter = [[2,3],[0,1]] do a product multiply
            ## [2,3] x [0,1] = [[2,0], [2,1], [3,0], [3,1]] 
            ## so we get the mapping for transfer
            ## 3D from 2x2x2 to 4x4x4
            ## [2,3] x [0,1] x [2,3] = [2,0,2], [2,0,3], [2,1,2], [2,1,3], ....
            
            to_iter = []
            for dd in range(self.graph_target.dimension):
                temp = []
                for pp in range(p_val):
                    temp.append(new_coor[dd] + pp * k_val)
                to_iter.append(temp)
                
            ### List all combinations to be replaced which is the product that has been explained before
            ## For example in 3d from 2 to 4
            ## old_coor (0,0,0), new coordinates = (0,0,0), (0,0,1), (0,1,0), (0,1,1) ....
            ## old_coor (1,1,1), new_coordinates = (2,2,2), (2,2,3), (2,3,2), .... 
            
            new_coordinates = []
            if self.graph_target.dimension == 1:
                new_coordinates = [[a] for a in to_iter[0]]
            else:
                for kk in to_iter:
                    if len(new_coordinates) == 0:
                        new_coordinates = kk 
                    else: 
                        new_coordinates = [list(cc[0] + [cc[1]]) if isinstance(cc[0], list) else list(cc)  for cc in list(itertools.product(new_coordinates, kk))]


            ## Replace all in the new coordinates with the base coordinates
            ## Connect to new hidden
            for coord in new_coordinates:
                quadrant = [int(c / self.graph_base.length) for c in coord]
                hid_pos = 0
                for ddd in range(self.graph_base.dimension):
                    hid_pos += quadrant[ddd] * (p_val  ** ddd)             

                target_point = self.graph_target._coordinate_to_point(coord)
                
                self.W_transfer[int(target_point), int(hid_pos * self.W_base.shape[1]) :int((hid_pos + 1) * self.W_base.shape[1])] = self.W_base[point, :]
           
        if k_val == 1:
            ind = np.array([[a, a] for a in range(0, self.bv_base.shape[1], 1)]).flatten()
            self.bv_transfer = self.bv_base[:,ind] 
            ind = np.array([[a, a] for a in range(0, self.bh_base.shape[1], 1)]).flatten()
            self.bh_transfer = self.bh_base[:,ind] 
        elif k_val == 2:
            ind = np.array([[a, a+1, a, a+1] for a in range(0, self.bv_base.shape[1], 2)]).flatten()
            self.bv_transfer = self.bv_base[:,ind] 
            ind = np.array([[a, a+1, a, a+1] for a in range(0, self.bh_base.shape[1], 2)]).flatten()
            self.bh_transfer = self.bh_base[:,ind] 
        elif k_val == 4:
            ind = np.array([[a, a+1, a+2, a+3, a, a+1, a+2, a+3] for a in range(0, self.bv_base.shape[1], 4)]).flatten()
            self.bv_transfer = self.bv_base[:,ind] 
            ind = np.array([[a, a+1, a+2, a+3, a, a+1, a+2, a+3] for a in range(0, self.bh_base.shape[1], 4)]).flatten()
            self.bh_transfer = self.bh_base[:,ind] 
        else:
            self.bv_transfer = np.concatenate((self.bv_base, self.bv_base), 1)  
            self.bh_transfer = np.concatenate((self.bh_base, self.bh_base), 1)  

        self.model_target.set_parameters([self.W_transfer, self.bv_transfer, self.bh_transfer]) 

        

    def tiling_sample(self, k_val):
        """ 
            Tiling sample for sample transfer
        """
        if hasattr(self.learner_base, 'samples_observ'):
            samples = self.learner_base.samples_observ.numpy()
        else:
            samples = self.learner_base.samples.numpy()
        multiplier = self.model_target.num_visible // self.model_base.num_visible

        ## cut paste
        if k_val == 0:
            tiled_samples = samples
        elif k_val == 1:
            ind = np.array([[a, a] for a in range(0, samples.shape[1], 1)]).flatten()
            tiled_samples = samples[:,ind] 
        elif k_val == 2:
            ind = np.array([[a, a+1, a, a+1] for a in range(0, samples.shape[1], 2)]).flatten()
            tiled_samples = samples[:,ind] 
        elif k_val == 4:
            ind = np.array([[a, a+1, a+2, a+3, a, a+1, a+2, a+3] for a in range(0, samples.shape[1], 4)]).flatten()
            tiled_samples = samples[:,ind] 
        else:
            tiled_samples = np.concatenate((samples, samples), 1)  

        return tiled_samples

    def decimate(self, k_val):
        """
            Decimate parameters to transfer from large to small system.
        """
        ## Decimate rows
        if k_val == 0:
            ind = np.arange(self.model_base.num_visible)
            np.random.shuffle(ind)
            ind = ind[:self.model_target.num_visible]
            W_temp = self.W_base[ind]
        elif k_val == 1:
            ind = np.array([[a] for a in range(0, self.model_base.num_visible , 2)]).flatten()
            W_temp = self.W_base[ind]
        elif k_val == 2:
            ind = np.array([[a, a+1] for a in range(0, self.model_base.num_visible, 4)]).flatten()
            W_temp = self.W_base[ind]
        else:
            W_temp = self.W_base[:k_val]
        
        #self.W_transfer = W_temp

        ## Sort hidden nodes by the maximum of the absolute value of the weight in a hidden node
        #sum_col = np.sum(np.abs(W_temp), 0)
        sum_col = np.max(np.abs(W_temp), 0)
        rank = np.argsort(sum_col)

        self.W_transfer = W_temp[:,rank[len(rank) // 2:]]

        ## Decimate bias
        if k_val == 0:
            ind = np.arange(self.bv_base.shape[1])
            np.random.shuffle(ind)
            ind = ind[:self.bv_base.shape[1] // 2]
            self.bv_transfer = np.reshape(self.bv_base[0, ind], (1, self.bv_base.shape[1] // 2))

            ind = np.arange(self.bh_base.shape[1])
            np.random.shuffle(ind)
            ind = ind[:self.bh_base.shape[1] // 2]
            self.bh_transfer = np.reshape(self.bh_base[0, ind], (1, self.bh_base.shape[1] // 2))
        if k_val == 1:
            ind = np.array([[a] for a in range(0, self.bv_base.shape[1] , 2)]).flatten()
            self.bv_transfer = np.reshape(self.bv_base[0, ind], (1, self.bv_base.shape[1] // 2))
            ind = np.array([[a] for a in range(0, self.bh_base.shape[1] , 2)]).flatten()
            self.bh_transfer = np.reshape(self.bh_base[0, ind], (1, self.bh_base.shape[1] // 2))
        elif k_val == 2:
            ind = np.array([[a, a+1] for a in range(0, self.bv_base.shape[1] , 4)]).flatten()
            self.bv_transfer = np.reshape(self.bv_base[0, ind], (1, self.bv_base.shape[1] // 2))
            ind = np.array([[a, a+1] for a in range(0, self.bh_base.shape[1] , 4)]).flatten()
            self.bh_transfer = np.reshape(self.bh_base[0, ind], (1, self.bh_base.shape[1] // 2))
        else:
            self.bv_transfer = np.reshape(self.bv_base[0, :self.bv_transfer.shape[1]], (1, self.bv_base.shape[1] // 2))
            self.bh_transfer = np.reshape(self.bh_base[0, :self.bh_transfer.shape[1]], (1, self.bh_base.shape[1] // 2))

        ## For keep hidden nodes the same
        #self.bh_transfer = self.bh_base

        self.model_target.set_parameters([self.W_transfer, self.bv_transfer, self.bh_transfer])


    def decimate_sample(self, k_val):
        """
            Decimate sample to transfer from large to small system.
        """
        if hasattr(self.learner_base, 'samples_observ'):
            samples = self.learner_base.samples_observ.numpy()
        else:
            samples = self.learner_base.samples.numpy()

        multiplier = self.model_base.num_visible // self.model_target.num_visible

        if k_val == 0:
            ind = np.arange(self.model_base.num_visible)
            np.random.shuffle(ind)
            ind = ind[:self.model_target.num_visible]
            tiled_samples = samples[:,ind]
        elif k_val == 1:
            ind = np.array([[a] for a in range(0, self.model_base.num_visible , 2)]).flatten()
            tiled_samples = samples[:,ind]
        elif k_val == 2:
            ind = np.array([[a, a+1] for a in range(0, self.model_base.num_visible, 4)]).flatten()
            tiled_samples = samples[:,ind]
        else:
            tiled_samples = samples[:,:k_val]


        return tiled_samples

