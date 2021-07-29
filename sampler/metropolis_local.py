from sampler import Sampler
import tensorflow as tf

class MetropolisLocal(Sampler):
    """
    This class is used to do a metropolis local sampling. 
    In metropolis local, the next sample is determined by flipping one random spin.
    """
    def __init__(self, num_samples, num_steps):
        """
        Construct a Metropolis Local sampler

        Args:
            num_samples: number of samples
            num_steps: number of steps in metropolis
        """
        Sampler.__init__(self, num_samples)
        self.num_steps = num_steps

    def get_initial_random_samples(self, sample_size, num_samples=None):
        """
            Get initial random samples with size [num_samples, sample_size]
            from a random uniform.
            Args:
                sample_size: the number of particles
                num_samples: number of samples
            Return:
                initial random samples
        """
        if num_samples is None:
            num_samples = self.num_samples

        init_data = tf.random.uniform((num_samples, sample_size), 0, 2, dtype=tf.dtypes.int32)
        init_data = tf.where(init_data == 0, -1, 1)
        
        return tf.cast(init_data, tf.dtypes.float32)

    def sample(self, model, initial_sample, num_samples):
        """
            Do a metropolis local sample from a given initial sample
            and model to get \Psi(x).
            Args:
                model: model to calculate \Psi(x)
                initial_sample: the initial sample
                num_samples: number of samples returned

            Return:
                new samples
        """
        sample = initial_sample

        for i in range(self.num_steps):
            sample = self.sample_once(model, sample, num_samples)
        return sample

    def sample_once(self, model, starting_sample, num_samples):
        """
            Do a one metropolis step from a given starting samples, model is used to calculate probability.

            Args:
                model: the model to calculate probability |\Psi(x)|^2
                starting_samples: the initial samples
                num_samples: number of samples returned
            Return:
                new samples from one metropolis local
        """
        ## Get new configuration by flipping the spin a random spin
        new_config = self.get_new_config(starting_sample, num_samples)

        ## Calculate the ratio of the new configuration and old configuration probability by computing |log(psi(x')) - log(psi(x))|^2
        if model.is_real():
            ratio = tf.abs(model.log_val_diff(new_config, starting_sample)) ** 2
        else:
            ratio = tf.abs(tf.exp(model.log_val_diff(new_config, starting_sample))) ** 2
        
        ## Sampling
        random = tf.random.uniform((num_samples, 1), 0, 1)

        ## Calculate acceptance
        accept = tf.squeeze(tf.greater(ratio, random))
        accept = tf.broadcast_to(tf.reshape(accept, (accept.shape[0], 1)), starting_sample.shape)

        ## Reject and accept samples
        sample = tf.where(accept, new_config, starting_sample)
        return sample

    def get_new_config(self, sample, num_samples):
        """
            Get a new configuration by flipping a random spin
            Args: 
                sample: the samples that want to be flipped randomly
                num_samples: the number of samples
            Return:
                new samples with a randomly flipped spin
        """
        num_points = int(sample.shape[1])

        row_indices = tf.reshape(tf.convert_to_tensor(range(num_samples)), (num_samples,1))
        col_indices = tf.random.uniform((num_samples, 1), 0, num_points, dtype=tf.dtypes.int32)
        indices = tf.concat([row_indices, col_indices], 1)

        elements = tf.gather_nd(sample, indices)
        old = tf.scatter_nd(indices, elements, (num_samples, num_points))
        new = tf.scatter_nd(indices, tf.negative(elements), (num_samples, num_points))
        return sample - old + new

    def get_all_samples(self, model, initial_sample, num_samples):
        all_samples = []
        sample = initial_sample
        for i in range(self.num_steps):
            sample = self.sample_once(model, sample, num_samples)
            all_samples.append(sample)

        return all_samples



    def to_xml(self):
        str = ""
        str += "<sampler>\n"
        str += "\t<type>metropolis_local</type>\n"
        str += "\t<params>\n"
        str += "\t\t<num_samples>%d</num_samples>\n" % self.num_samples
        str += "\t\t<num_steps>%d</num_steps>\n" % self.num_steps
        str += "\t</params>\n"
        str += "</sampler>\n"
        return str
