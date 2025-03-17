import tensorflow as tf
import tf_keras as keras

LAYER_SIZES = [600, 400, 200, 100]

class Encoder(keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.dense_layers = [keras.layers.Dense(x) for x in LAYER_SIZES]
    
    @tf.function
    def call(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        return x


class Decoder(keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        # Don't include the first layer, as it's now the input
        included_layers = LAYER_SIZES[1:].copy()
        included_layers.reverse()
        self.dense_layers = [keras.layers.Dense(x) for x in included_layers]
    
    @tf.function
    def call(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        return x


class AutoEncoder(keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    @tf.function
    def call(self, input):
        enc = self.encoder(input)
        return self.decoder(enc)
    

def generate_model():
    model = AutoEncoder()
    model.compile()
    
generate_model()