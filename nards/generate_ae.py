import tensorflow as tf
import tf_keras as keras
import pickle


FILTERS = [128, 64, 32]

LATENT_DIM = 150

class Encoder(keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_layers = [keras.layers.Conv2D(x, (3, 3), padding="same", activation="relu") for x in FILTERS]
        self.flatten = keras.layers.Flatten()
        self.latent = keras.layers.Dense(LATENT_DIM, activation="relu")


    @tf.function
    def call(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        print("Shape of x: {}".format(x.shape))
        x = self.flatten(x)
        print("Len after flatten {}".format(x.shape))
        return self.latent(x)


class Decoder(keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dense_input = keras.layers.Dense(8 * 8 * 32)
        included_layers = FILTERS.copy()
        included_layers.reverse()
        self.conv_layers = [keras.layers.Conv2D(x, (3, 3), padding="same", activation="relu") for x in included_layers]
        self.output_layer = keras.layers.Conv2D(12, (3, 3), padding="same", activation="sigmoid")


    @tf.function
    def call(self, x):
        x = self.dense_input(x)
        x = tf.reshape(x, (-1, 8, 8, FILTERS[-1]))
        for layer in self.conv_layers:
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
    model.compile(optimizer, loss="binary_crossentropy", metrics=["accuracy", "mse"])
    return model
    

def prepare_moves(moves):
    for move in moves:
        data = tf.reshape(tf.constant(move["board"]["data"]), (8, 8, 12))
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
    output_signature = (tf.TensorSpec(shape=(8, 8, 12), dtype=tf.float32), tf.TensorSpec(shape=(8, 8, 12), dtype=tf.float32))
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature).shuffle(100000).batch(32)
    # dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs)).shuffle(70000).batch(32)
    train_dataset, test_dataset = tf.keras.utils.split_dataset(dataset, left_size=0.9)
    # model.fit(train_dataset, epochs=50, validation_data=test_dataset, validation_freq=5)
    model.fit(train_dataset, epochs=10)
    
    model.evaluate(test_dataset)
    
    input_spec = tf.TensorSpec(shape=(None, 8, 8, 12), dtype=tf.uint64)
    signatures = { "call": model.call.get_concrete_function(input_spec) }
    model.save("ae_model", save_format="tf", signatures=signatures)
    
    signatures = { "call": model.encoder.call.get_concrete_function(input_spec) }
    model.encoder.save("enc_model", save_format="tf", signatures=signatures)
