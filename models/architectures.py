import gin
import tensorflow as tf


@gin.configurable
class LSTM_Autoencoder:
    '''
    A class for building an LSTM-based autoencoder for time-series data.

        Attributes:
            time_steps (int): Number of time steps in input sequences.
            input_dim (int): Dimensionality of each time step.
            autoencoder_units (list): List of integers specifying the number of LSTM units in each layer.
            dropout_rate (float): Dropout rate applied after each LSTM layer.
    '''
    def __init__(self, time_steps, input_dim, autoencoder_units, dropout_rate):
        self.time_steps = time_steps
        self.input_dim = input_dim
        self.autoencoder_units = autoencoder_units
        self.dropout_rate = dropout_rate
        
    def build_model(self,):
        '''
        Builds the LSTM Autoencoder model.

        Returns:
            tf.keras.Model: A compiled LSTM autoencoder model.
        '''
        # Encoder
        inputs = tf.keras.layers.Input(shape=(self.time_steps, self.input_dim))
        encoded = inputs
        for units in self.autoencoder_units:
            encoded = tf.keras.layers.LSTM(units ,activation='tanh', return_sequences=(units != self.autoencoder_units[-1]))(encoded)
            encoded = tf.keras.layers.Dropout(self.dropout_rate)(encoded)

        # Decoder
        decoded = tf.keras.layers.RepeatVector(self.time_steps)(encoded)
        for units in self.autoencoder_units[::-1]:
            decoded = tf.keras.layers.LSTM(units ,activation='tanh', return_sequences=True)(decoded)
            decoded = tf.keras.layers.Dropout(self.dropout_rate)(decoded)

        outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.input_dim))(decoded)
        model = tf.keras.models.Model(inputs, outputs)

        return model
