import tensorflow as tf
import tf_keras as keras
import pickle
import generate_ae
import sys
import random

FILTERS = [256, 128, 64, 32]
UNITS = [200, 100, 32]


class NardsModel(keras.Model):
    def __init__(self, encoder):
        super(NardsModel, self).__init__()
        self.encoder = encoder
        self.feature_extraction = []
        for unit in [128, 64]:
            self.feature_extraction.append(keras.layers.Dense(unit, activation="relu"))
            self.feature_extraction.append(keras.layers.Dropout(0.2))
        self.concat = keras.layers.Concatenate()
        # self.dense_input = keras.layers.Dense(8 * 8 * generate_ae.FILTERS[-1])
        self.dense_and_dropout = []
        for unit in UNITS:
            self.dense_and_dropout.append(keras.layers.Dense(unit, activation="relu"))
            self.dense_and_dropout.append(keras.layers.Dropout(0.2))
        # self.dropout_input = keras.layers.Dropout(0.2)
        # self.conv_layers = [keras.layers.Conv2D(x, (3, 3), padding="same", activation="relu", kernel_regularizer=keras.regularizers.L2(0.0001)) for x in FILTERS]
        # self.flatten = keras.layers.Flatten()
        # self.dropout_output = keras.layers.Dropout(0.2)
        self.output_layer = keras.layers.Dense(2, activation="sigmoid")

    @tf.function()
    def call(self, lhs, rhs):
        lhs = self.encoder(lhs)
        for layer in self.feature_extraction:
            lhs = layer(lhs)
        rhs = self.encoder(rhs)
        for layer in self.feature_extraction:
            rhs = layer(rhs)
        x = self.concat([lhs, rhs])
        # x = self.dense_input(x)
        # x = self.dropout_input(x)
        # Reshape from the latent space
        # x = tf.reshape(x, (-1, 8, 8, generate_ae.FILTERS[-1]))
        for layer in self.dense_and_dropout:
            x = layer(x)
        # x = self.dropout_output(x)
        # x = self.flatten(x)
        return self.output_layer(x)


def prepare_moves(moves):
    for move in moves:
        input = tf.reshape(tf.constant(move["board"]["data"]), (8, 8, 12))
        output = tf.constant(move["result"])
        yield input, output
        

def prepare_file(file_name):
    parsed = []
    data = pickle.load(open(file_name, "rb"))
    for position in data:
        parsed.append(tf.reshape(tf.constant(position["board"]["data"]), (8, 8, 12)))
    return tf.convert_to_tensor(parsed)


def data_generator(self):
    while True:
        first_index = random.randint(0, 3)
        if first_index > 1:
            second_index = first_index - 2
        else:
            second_index = first_index + 2
        if first_index == 0 or first_index == 3:
            won = tf.constant([1, 0])
        else:
            won = tf.constant([0, 1])
        lhs_game_index = random.randint(0, len(self.files[first_index] - 1))
        rhs_game_index = random.randint(0, len(self.files[second_index] - 1))
        yield (self.files[first_index][lhs_game_index], self.files[second_index][rhs_game_index]), won


def train_model(model):
    filenames = ["white_won", "white_lost", "black_lost", "black_won"]
    files = [prepare_file(p) for p in filenames]

    # with open("latest.pickle", "rb") as f:
        # moves = pickle.load(f)
        # generator = lambda: prepare_moves(moves)
    output_signature = ((tf.TensorSpec(shape=(8, 8, 12), dtype=tf.float32), tf.TensorSpec(shape=(8, 8, 12), dtype=tf.float32)), tf.TensorSpec(shape=(2,), dtype=tf.float32))
    # dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature).shuffle(100000).batch(16)
    dataset = tf.data.Dataset.from_generator(data_generator, output_signature=output_signature, args=files).take(100_000).batch(16)
    train_dataset, test_dataset = tf.keras.utils.split_dataset(dataset, left_size=0.9)
    model.fit(train_dataset, epochs=5)
    
    model.evaluate(test_dataset)
    
    input_spec = (tf.TensorSpec(shape=(None, 8, 8, 12), dtype=tf.float32), tf.TensorSpec(shape=(None, 8, 8, 12), dtype=tf.float32))
    signatures = { "call": model.call.get_concrete_function(input_spec) }
    model.save("ae_model", save_format="tf", signatures=signatures)


# seq = PositionSequence()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "new":
        model = generate_ae.generate_model()
        generate_ae.train_model(model)
        encoder = model.encoder
    else:
        encoder = keras.models.load_model("enc_model")
    encoder.trainable = False
    
    n_model = NardsModel(encoder)
    optimizer = keras.optimizers.Adam(0.001)
    accuracy_metric = keras.metrics.BinaryAccuracy(threshold=0.5)
    n_model.compile(optimizer, loss="binary_crossentropy", metrics=[accuracy_metric, "mae"])
    train_model(n_model)