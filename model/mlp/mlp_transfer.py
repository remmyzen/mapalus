import tensorflow as tf
import copy
import sys
import numpy as np
import os
import pickle
import itertools

class MLPTransfer(object):
    """
    Handle the transfer for multilayer perceptron model
    """
    
    def __init__ (self, model_target, graph_target, base_model_path, base_model_number=None, divide=1., index_layer=None):
        """
        Initialize an MLP transfer object.
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
        self.index_layer = index_layer
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
        self.params_base = [a / self.divide for a in self.model_base.get_parameters()]
        self.params_target = self.model_target.get_parameters()


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

    def cutpaste_mapping(self, mapping):
        """
        If the base model and target model is the same, we just cut and paste the parameters for the transfer.
        However, this method needs a mapping of which parameter to which parameter
        A mapping of {0:1} means that parameter index 0 of target network will be mapped to parameter index 1 of the base network.
        """
        assert self.model_target.num_visible >= self.model_base.num_visible, "Number of visible node in the model must be larger than or equal to the numbero f visible node in the base model!"

        self.new_params = []

        for ii, tra_weight in enumerate(self.params_target):
            if ii in mapping.keys():
                self.new_params.append(self.params_base[mapping[ii]])
            else:
                self.new_params.append(tra_weight)
        
        self.model_target.set_parameters(self.new_params) 


    def tiling (self, k_val):
        """
        Not general enough, only works for specified k 
        """
        assert self.model_target.num_visible >= self.model_base.num_visible and self.model_target.num_visible % self.model_base.num_visible == 0, "Number of visible node in the model must be larger than or equal to and divisible by the number of visible node in the base model!"
        #assert self.graph_base.length % k_val == 0, "k must be divisible by the number of visible node in base model!"
        
        p_val = self.graph_target.length / self.graph_base.length

        self.new_params = []
        ### (0,2)-tiling similar to cut and paste
        if k_val == 0:
            ## Loop through all weights
            for base_weight, tra_weight in zip(self.params_base, self.params_target):
                ## If the weight dimension is the same just transfer directly
                if base_weight.shape == tra_weight.shape:
                    self.new_params.append(base_weight)

                ## If the row is changed, just copy the part of the weights old weights
                elif len(base_weight.shape) == 1 or base_weight.shape[1] == tra_weight.shape[1]:
                    half_row = tra_weight.shape[0] // 2
                    new_arr = tra_weight
                    new_arr[:base_weight.shape[0]] = base_weight
                    self.new_params.append(new_arr)

                ## If the col is changed, just repeat it
                elif base_weight.shape[0] == tra_weight.shape[0]:
                    half_col = tra_weight.shape[1] // 2
                    new_arr = tra_weight
                    new_arr[:, :base_weight.shape[1]] = base_weight
                    self.new_params.append(new_arr)
                ## If the both the row and column is changed, do the proper transfer by connecting only to the new weights
                else:
                    multiplier = tra_weight.shape[0] // base_weight.shape[0]
                    half_row = tra_weight.shape[0] // 2
                    half_col = tra_weight.shape[1] // 2
                    new_arr = tra_weight
                    new_arr[:half_row, :half_col] = base_weight
                    self.new_params.append(new_arr)            
            
        ### (1,2)-tiling
        elif k_val == 1:
            ## Loop through all weights
            for base_weight, tra_weight in zip(self.params_base, self.params_target):
                ## If the weight dimension is the same just transfer directly
                if base_weight.shape == tra_weight.shape:
                    self.new_params.append(base_weight)

                ## If the row is changed, just repeat it
                elif len(base_weight.shape) == 1 or base_weight.shape[1] == tra_weight.shape[1]:
                    multiplier = tra_weight.shape[0] // base_weight.shape[0]
                    self.new_params.append(np.repeat(base_weight, multiplier, 0))

                ## If the col is changed, just repeat it
                elif base_weight.shape[0] == tra_weight.shape[0]:
                    multiplier = tra_weight.shape[1] // base_weight.shape[1]
                    self.new_params.append(np.repeat(base_weight, multiplier, 1))
                ## If the both the row and column is changed, do the proper transfer by connecting only to the new weights
                else:
                    multiplier = tra_weight.shape[0] // base_weight.shape[0]
                    half_row = tra_weight.shape[0] // 2
                    half_col = tra_weight.shape[1] // 2
                    repeat = np.repeat(base_weight,  multiplier, 0)
                    new_arr = tra_weight
                    new_arr[:half_row, :half_col] = repeat[:half_row,:]
                    new_arr[half_row:, half_col:] = repeat[half_row:,:]

                    self.new_params.append(new_arr)            

        ### (2,2)-tiling
        elif k_val == 2:
            ## Loop through all weights
            for base_weight, tra_weight in zip(self.params_base, self.params_target):
                ## Create a pattern [1, 2, 1, 2, 3, 4, 3, 4 ...] for indexing
                ind = np.array([[a, a+1, a, a+1] for a in range(0,len(base_weight),2)]).flatten() 

                ## If the weight dimension is the same just transfer directly
                if base_weight.shape == tra_weight.shape:
                    self.new_params.append(base_weight)

                ## If the row is changed, just tile it also handle bias
                elif len(base_weight.shape) == 1 or base_weight.shape[1] == tra_weight.shape[1]:
                    ind = np.array([[a, a+1, a, a+1] for a in range(0,base_weight.shape[0],2)]).flatten() 
                    self.new_params.append(base_weight[ind])

                ## If the col is changed, just tile it
                elif base_weight.shape[0] == tra_weight.shape[0]:
                    ind = np.array([[a, a+1, a, a+1] for a in range(0,base_weight.shape[1],2)]).flatten() 
                    self.new_params.append(base_weight[:, ind])
                ## If the both the row and column is changed, do the proper transfer by connecting only to the new weights
                else:
                    multiplier = tra_weight.shape[0] // base_weight.shape[0]
                    half_row = tra_weight.shape[0] // 2
                    half_col = tra_weight.shape[1] // 2
                    new_arr = tra_weight
                    new_arr[:half_row, :half_col] = base_weight[ind[:len(ind) // 2]]
                    new_arr[half_row:, half_col:] = base_weight[ind[len(ind) // 2:]]

                    self.new_params.append(new_arr)            
        ### (4,2)-tiling
        #elif k_val == 4:
        #    ## Loop through all weights
        #    for base_weight, tra_weight in zip(self.params_base, self.params_target):
        #        ## Create a pattern [1, 2, 3, 4, 1, 2, 3, 4, ...] for indexing
        #        ind = np.array([[a, a+1, a+2, a+3, a, a+1, a+2, a+3] for a in range(0,len(base_weight),4)]).flatten() 

        #        ## If the weight dimension is the same just transfer directly
        #        if base_weight.shape == tra_weight.shape:
        #            self.new_params.append(base_weight)

        #        ## If the row is changed, just tile it also handle bias
        #        elif len(base_weight.shape) == 1 or base_weight.shape[1] == tra_weight.shape[1]:
        #            ind = np.array([[a, a+1, a+2, a+3, a, a+1, a+2, a+3] for a in range(0,base_weight.shape[0],4)]).flatten() 
        #            self.new_params.append(base_weight[ind])

        #        ## If the col is changed, just tile it
        #        elif base_weight.shape[0] == tra_weight.shape[0]:
        #            ind = np.array([[a, a+1, a+2, a+3, a, a+1, a+2, a+3] for a in range(0,base_weight.shape[1],4)]).flatten() 
        #            self.new_params.append(base_weight[:, ind])
        #        ## If the both the row and column is changed, do the proper transfer by connecting only to the new weights
        #        else:
        #            multiplier = tra_weight.shape[0] // base_weight.shape[0]
        #            half_row = tra_weight.shape[0] // 2
        #            half_col = tra_weight.shape[1] // 2
        #            new_arr = tra_weight
        #            new_arr[:half_row, :half_col] = base_weight[ind[:len(ind) // 2]]
        #            new_arr[half_row:, half_col:] = base_weight[ind[len(ind) // 2:]]

        #            self.new_params.append(new_arr)            
        ### (L,2)-tiling
        else:
            ## Loop through all weights
            for ii, (base_weight, tra_weight) in enumerate(zip(self.params_base, self.params_target)):

                if self.index_layer is not None and ii not in self.index_layer:
                    self.new_params.append(tra_weight)
                    continue

                ## If the weight dimension is the same just transfer directly
                if base_weight.shape == tra_weight.shape:
                    self.new_params.append(base_weight)

                ## If the row is changed, just tile it also handle bias
                elif len(base_weight.shape) == 1 or base_weight.shape[1] == tra_weight.shape[1]:
                    multiplier = tra_weight.shape[0] // base_weight.shape[0]
                    ## handle bias
                    if len(base_weight.shape) == 1:
                        self.new_params.append(np.tile(base_weight, multiplier))
                    else:
                        self.new_params.append(np.tile(base_weight, (multiplier, 1)))

                ## If the col is changed, just tile it
                elif base_weight.shape[0] == tra_weight.shape[0]:
                    multiplier = tra_weight.shape[1] // base_weight.shape[1]
                    self.new_params.append(np.tile(base_weight, (1, multiplier)))
                ## If the both the row and column is changed, do the proper transfer by connecting only to the new weights
                else:
                    multiplier = tra_weight.shape[0] // base_weight.shape[0]
                    half_row = tra_weight.shape[0] // 2
                    half_col = tra_weight.shape[1] // 2
                    new_arr = tra_weight
                    new_arr[:half_row, :half_col] = base_weight
                    new_arr[half_row:, half_col:] = base_weight

                    self.new_params.append(new_arr)            

        self.model_target.set_parameters(self.new_params) 
        

    def tiling_sample(self, k_val):
        """
        Tile the sample according to k_val.
        """

        if hasattr(self.learner_base, 'samples_observ'):
            if tf.is_tensor(self.learner_base.samples_observ):
                samples = self.learner_base.samples_observ.numpy()
            else:
                samples = self.learner_base.samples_observ
        else:
            if tf.is_tensor(self.learner_base.samples):
                samples = self.learner_base.samples.numpy()
            else:
                samples = self.learner_base.samples

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
