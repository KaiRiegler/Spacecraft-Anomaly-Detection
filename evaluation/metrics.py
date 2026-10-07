import tensorflow as tf


class Metrics():
    '''
    A class to compute and store various evaluation metrics. 
        
        Attributes:
                name (str): Name prefix for the metric group.
    '''
    def __init__(self, name):
        self.name = name
        self.metrics = {"loss": tf.keras.metrics.Mean(name=self.name+'_loss')}

    def update_state(self, loss):
        '''
        Updates the loss metric.

            Parameters:
                    loss (tf.Tensor or float): The computed loss value.
        '''
        self.metrics["loss"].update_state(loss)

    def result(self, metric):
        '''
        Retrieves the computed value for a specific metric.

            Parameters:
                    metric (str): Name of the metric.

            Returns:
                    tf.Tensor: Computed metric value.
        '''
        return self.metrics[metric].result()
    
    def reset_states(self):
        '''
        Resets all tracked metrics.
        '''
        for m in self.metrics.values():
            m.reset_states()
    