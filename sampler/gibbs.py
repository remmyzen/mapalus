from sampler import Sampler
import tensorflow as tf


class Gibbs(Sampler):
    """
    This class is used to do a Gibbs sampling. 
    Gibbs sampling is a special case of Metropolis algorithm where acceptance ratio is 1.
    Only works for RBM machine with real positive wave function
    """

    def __init__(self, num_samples, num_steps=1):
        """
        Construct a Gibbs sampler

        Args:
            num_samples: number of samples
            num_steps: number of steps (1 step = sample h from p(h| v) and sample v from p (v|h)
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
            Do a gibbs sampling from a given initial sample
            by sampling from p(v | h) from the RBM
            Args:
                model: model to calculate \Psi(x)
                initial_sample: the initial sample
                num_samples: number of samples returned

            Return:
                new samples
        """
        sample = initial_sample
        for i in range(self.num_steps):
            sample = model.get_new_visible(sample)

        return sample

    def get_all_samples(self, model, initial_sample, num_samples):
        """ 
            Get sample from gibbs sampling
        """ 
        all_samples = []
        sample = initial_sample
        for i in range(self.num_steps):
            sample = model.get_new_visible(sample)
            all_samples.append(sample)

        return all_samples

    def to_xml(self):
        str = ""
        str += "<sampler>\n"
        str += "\t<type>gibbs</type>\n"
        str += "\t<params>\n"
        str += "\t\t<num_samples>%d</num_samples>\n" % self.num_samples
        str += "\t\t<num_steps>%d</num_steps>\n" % self.num_steps
        str += "\t</params>\n"
        str += "</sampler>\n"
        return str
