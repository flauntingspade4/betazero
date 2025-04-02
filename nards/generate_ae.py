import tensorflow as tf
import tf_keras as keras
import pickle
import random


FILTERS = [256, 256, 128]
FILTERS = [256, 256, 128]

LATENT_DIM = 100

@keras.saving.register_keras_serializable(package="nards")
class Encoder(keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_layers = [keras.layers.Conv2D(x, (3, 3), padding="same", activation="relu") for x in FILTERS]
        # self.conv_layers = [keras.layers.Dense(x, activation="relu") for x in FILTERS]
        # self.conv_layers = [keras.layers.Dense(x, activation="relu") for x in FILTERS]
        self.flatten = keras.layers.Flatten()
        self.latent = keras.layers.Dense(LATENT_DIM, activation="relu")


    @tf.function
    def call(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        return self.latent(x)
    
    """def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            "filters": FILTERS,
            "latent_dim": LATENT_DIM,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls()"""
    

class Decoder(keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dense_input = keras.layers.Dense(8 * 8 * FILTERS[-1])
        included_layers = FILTERS.copy()
        included_layers.reverse()
        # print(included_layers)
        self.conv_layers = [keras.layers.Conv2D(x, (3, 3), padding="same", activation="relu") for x in included_layers]
        # self.conv_layers = [keras.layers.Dense(x, activation="relu") for x in included_layers]
        # self.output_layer = keras.layers.Conv2D(12, (3, 3), padding="same", activation="sigmoid")
        self.output_layer = keras.layers.Dense(12, activation="sigmoid")


    @tf.function
    def call(self, x):
        x = self.dense_input(x)
        # x = tf.reshape(x, (-1, 8, 8, FILTERS[-1]))
        for layer in self.conv_layers:
            x = layer(x)
        return tf.reshape(self.output_layer(x), (-1, 8, 8, 12))


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
    optimizer = keras.optimizers.Adam(learning_rate=0.00005)
    accuracy_metric = keras.metrics.BinaryAccuracy(threshold=0.5)
    model.compile(optimizer, loss="binary_crossentropy", metrics=[accuracy_metric, "mse", "mae"])
    return model
    

def prepare_moves(moves):
    while True:
        index = random.randrange(len(moves))
        yield moves[index], moves[index]


def train_model(model):
    moves = []
    for name in ["lostsmall.pickle", "wonsmall.pickle"]:
        n_moves = pickle.load(open(name, "rb"))
        for m in n_moves:
            moves.append(tf.reshape(tf.constant(m["board"]["data"]), (8, 8, 12)))
            if len(n_moves) > 800_000:
                break
    print("{} total moves".format(len(moves)))
    output_signature = (tf.TensorSpec(shape=(8, 8, 12), dtype=tf.float32), tf.TensorSpec(shape=(8, 8, 12), dtype=tf.float32))
    dataset = tf.data.Dataset.from_generator(prepare_moves, output_signature=output_signature, args=[moves]).take(100_000).batch(16)
    # train_dataset, test_dataset = tf.keras.utils.split_dataset(dataset, left_size=0.9)
    model.fit(dataset, epochs=20)
    
    # model.evaluate(test_dataset)
    # model.evaluate(test_dataset)
    
    input_spec = tf.TensorSpec(shape=(None, 8, 8, 12), dtype=tf.float32)
    signatures = { "call": model.call.get_concrete_function(input_spec) }
    model.save("ae_model", save_format="tf", signatures=signatures)
    
    signatures = { "call": model.encoder.call.get_concrete_function(input_spec) }
    model.encoder.save("enc_model", save_format="tf", signatures=signatures)
    return model


if __name__ == "__main__":
    # if len(sys.argv) > 1 and sys.argv[1] == "-n":
    model = generate_model()
    train_model(model)
    
    input_spec = tf.TensorSpec(shape=(None, 8, 8, 12), dtype=tf.float32)
    signatures = { "call": model.call.get_concrete_function(input_spec) }
    model.save("ae_model", save_format="tf", signatures=signatures)


