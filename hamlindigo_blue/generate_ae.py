import tensorflow as tf
import tf_keras as keras
import pickle
from os import listdir
from os.path import isfile, join
import random
import matplotlib.pyplot as plt


FILTERS = [256, 256, 128]
LATENT_DIM = 100


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
        x = self.flatten(x)
        return self.latent(x)


class Decoder(keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dense_input = keras.layers.Dense(8 * 8 * FILTERS[-1])
        included_layers = FILTERS.copy()
        included_layers.reverse()
        self.conv_layers = [keras.layers.Conv2D(x, (3, 3), padding="same", activation="relu") for x in included_layers]
        self.output_layer = keras.layers.Dense(14, activation="sigmoid")


    @tf.function
    def call(self, x):
        x = self.dense_input(x)
        x = tf.reshape(x, (-1, 8, 8, FILTERS[-1]))
        for layer in self.conv_layers:
            x = layer(x)
        return tf.reshape(self.output_layer(x), (-1, 8, 8, 14))


class AutoEncoder(keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()


    @tf.function
    def call(self, input):
        enc = self.encoder(input)
        return self.decoder(enc)


def plot_losses(losses, y_label):
    epochs = range(0, len(losses) + 0)  # Epoch numbers start from 1
    ticks = range(0, len(losses) + 1, 5)
    plt.rc('font', size=30)

    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    plt.plot(epochs, losses, marker='o', linestyle='-', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.xticks(ticks)
    plt.tight_layout()
    plt.show()

def generate_model():
    model = AutoEncoder()
    optimizer = keras.optimizers.Adam(learning_rate=0.00005)
    accuracy_metric = keras.metrics.BinaryAccuracy(threshold=0.5)
    model.compile(optimizer, loss="binary_crossentropy", metrics=[accuracy_metric, "mse", "mae"])
    return model
    

def prepare_moves(white_won, black_won):
    while True:
        if bool(random.getrandbits(1)):
            chosen = random.choice(random.choice(white_won))
        else:
            chosen = random.choice(random.choice(black_won))
        yield chosen, chosen

        
def prepare_file(file_name):
    parsed = []
    data = pickle.load(open(file_name, "rb"))
    for i, position in enumerate(data):
        parsed.append(tf.reshape(tf.constant(position["data"], dtype=tf.float32), (8, 8, 14)))
    return parsed


def train_model(model):
    white_won_files = [join("white_won", f) for f in listdir("white_won") if isfile(join("white_won", f))]
    black_won_files = [join("black_won", f) for f in listdir("black_won") if isfile(join("black_won", f))]
    
    losses = []

    output_signature = (tf.TensorSpec(shape=(8, 8, 14), dtype=tf.float32), tf.TensorSpec(shape=(8, 8, 14), dtype=tf.float32))
    for _ in range(5):
        chosen_white = [prepare_file(c) for c in random.choices(white_won_files, k=2)]
        chosen_black = [prepare_file(c) for c in random.choices(black_won_files, k=2)]
        dataset = tf.data.Dataset.from_generator(prepare_moves, output_signature=output_signature, args=[chosen_white, chosen_black]).batch(32).take(2000)
        
        history = model.fit(dataset, epochs=4)
        for l in history.history["loss"]:
            losses.append(l)
        
    signatures = { "call": model.encoder.call.get_concrete_function(tf.TensorSpec(shape=(None, 8, 8, 14), dtype=tf.float32)) }
    model.encoder.save("enc_model", save_format="tf", signatures=signatures)
    plot_losses(losses, "Autoencoder Training Loss")
    return model


if __name__ == "__main__":
    model = generate_model()
    train_model(model)

