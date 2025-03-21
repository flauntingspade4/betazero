import tensorflow as tf
import tf_keras as keras
import sys
import pickle


LAYER_SIZES = [600, 400, 300, 200, 150]

class Encoder(keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.dense_layers = [keras.layers.Dense(x, activation="relu") for x in LAYER_SIZES]


    @tf.function
    def call(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        return x


class Decoder(keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        # Don't include the first layer, as it's now the input
        included_layers = LAYER_SIZES[:-1].copy()
        included_layers.reverse()
        self.dense_layers = [keras.layers.Dense(x, activation="relu") for x in included_layers]
        self.output_layer = keras.layers.Dense(8 * 8 * 12, activation="sigmoid")


    @tf.function
    def call(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        return self.output_layer(x)


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
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer, loss="mse", metrics="accuracy")
    return model
    

def prepare_moves(moves):
    for move in moves:
        data = tf.constant(move["board"]["data"])
        # inputs.append(data)
        # outputs.append(data)
        yield data, data


if __name__ == "__main__":
    # if len(sys.argv) > 1 and sys.argv[1] == "-n":
    model = generate_model()

    with open("latest.pickle", "rb") as f:
        # inputs, outputs = [], []
        moves = pickle.load(f)
        # prepare_moves(moves, inputs, outputs)
        generator = lambda: prepare_moves(moves)
    output_signature = (tf.TensorSpec(shape=(8 * 8 * 12), dtype=tf.float32), tf.TensorSpec(shape=(8 * 8 * 12), dtype=tf.float32))
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature).shuffle(100000).batch(32)
    # dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs)).shuffle(70000).batch(32)
    train_dataset, test_dataset = tf.keras.utils.split_dataset(dataset, left_size=0.9)
    # model.fit(train_dataset, epochs=50, validation_data=test_dataset, validation_freq=5)
    model.fit(train_dataset, epochs=10)
    
    model.evaluate(test_dataset)
    
    input_spec = tf.TensorSpec(shape=(None, 8 * 8 * 12), dtype=tf.uint64)
    signatures = { "call": model.call.get_concrete_function(input_spec) }
    model.save("ae_model", save_format="tf", signatures=signatures)
    
    signatures = { "call": model.encoder.call.get_concrete_function(input_spec) }
    model.encoder.save("enc_model", save_format="tf", signatures=signatures)
