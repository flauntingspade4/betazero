import tensorflow as tf
import tf_keras as keras
import sys
import pickle


LAYER_SIZES = [600, 400, 200, 100]

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
        # inputs.append(move["board"]["data"])
        # outputs.append(move["board"]["data"])
        data = tf.constant(move["board"]["data"])
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
    # print("{} inputs and {} outputs (Should be equal)".format(len(inputs), len(outputs)))
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature).shuffle(50000).batch(5)
    for d in dataset:
        print(d)
        break
    print("Made dataset")
    model.fit(dataset, epochs=50)