import gin
import tensorflow as tf


@gin.configurable
def augment(signal):
    '''
    Applies random augmentations to a given signal.

        Parameters:
                signal (tf.Tensor): Input signal tensor.

        Returns:
                tf.Tensor: Augmented signal tensor.
    '''
    def add_noise(signal, noise_level=0.01):
        '''
        Adds random Gaussian noise to the signal.

            Parameters:
                    signal (tf.Tensor): Input signal tensor.
                    noise_level (float, optional): Standard deviation of the noise. Defaults to 0.01.

            Returns:
                    tf.Tensor: Noisy signal tensor.
        '''
        noise = tf.random.normal(tf.shape(signal), mean=0.0, stddev=noise_level, dtype=signal.dtype)
        return signal + noise

    def time_shift(signal, shift):
        '''
        Shifts the signal by a given number of samples.

            Parameters:
                    signal (tf.Tensor): Input signal tensor.
                    shift (int): Number of samples to shift the signal.

            Returns:
                    tf.Tensor: Time-shifted signal tensor.
        '''
        return tf.roll(signal, shift, axis=0)

    def scale(signal):
        '''
        Applies a small random scaling factor to the signal.

            Parameters:
                    signal (tf.Tensor): Input signal tensor.

            Returns:
                    tf.Tensor: Scaled signal tensor.
        '''
        return signal * (tf.random.uniform(tf.shape(signal), minval=0.99, maxval=1.01,dtype=signal.dtype))

    # Apply multiple augmentations
    signal = add_noise(signal, noise_level=0.02)
    signal = time_shift(signal, shift=tf.random.uniform(shape=(), minval=0, maxval=50,dtype=tf.int32))
    signal = scale(signal)
    
    return signal
