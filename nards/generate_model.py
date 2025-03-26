import tensorflow as tf
import tf_keras as keras
import pickle
import generate_ae
import sys
import random

UNITS = [256, 128, 64]
leaky = keras.layers.LeakyReLU()
L2_REGULARIZATION = 0.0001


class NardsModel(keras.Model):
    def __init__(self):
        super(NardsModel, self).__init__()
        # self.encoder = encoder
        self.feature_extraction = []
        for unit in [256, 256, 128]:
            self.feature_extraction.append(keras.layers.Conv2D(unit, (3, 3), padding="same", activation=leaky, kernel_regularizer=keras.regularizers.L2(L2_REGULARIZATION)))
            # self.feature_extraction.append(keras.layers.Dropout(0.2))
        self.concat = keras.layers.Concatenate()
        self.flatten = keras.layers.Flatten()
        # self.dense_input = keras.layers.Dense(8 * 8 * generate_ae.FILTERS[-1])
        self.dense_and_dropout = []
        for unit in UNITS:
            self.dense_and_dropout.append(keras.layers.Dense(unit, activation=leaky, kernel_regularizer=keras.regularizers.L2(L2_REGULARIZATION)))
            self.dense_and_dropout.append(keras.layers.Dropout(0.2))
        # self.dropout_input = keras.layers.Dropout(0.2)
        # self.conv_layers = [keras.layers.Conv2D(x, (3, 3), padding="same", activation="relu", kernel_regularizer=keras.regularizers.L2(0.0001)) for x in FILTERS]
        self.output_layer = keras.layers.Dense(2, activation="softmax")

    @tf.function()
    def call(self, inputs):
        lhs, rhs = inputs
        for layer in self.feature_extraction:
            lhs = layer(lhs)
            rhs = layer(rhs)
        x = self.concat([lhs, rhs])
        x = self.flatten(x)
        # x = self.dense_input(x)
        # x = self.dropout_input(x)
        # Reshape from the latent space
        for layer in self.dense_and_dropout:
            x = layer(x)
        return self.output_layer(x)


def prepare_file(file_name, parsed):
    with open(file_name, "rb") as f:
        data = pickle.load(f)
        print("Doing file with {} entries".format(len(data)))
        for i, position in enumerate(data):
            parsed.append(tf.reshape(tf.constant(position["board"]["data"], dtype=tf.float32), (8, 8, 12)))
            if i > 200_000:
                break
        print("Done file")


def data_generator(files):
    while True:
        first_index = random.randint(0, 1)
        second_index = 1 - first_index
        if first_index == 0:
            won = tf.constant([1.0, 0.0], dtype=tf.float32)
        else:
            won = tf.constant([0.0, 1.0], dtype=tf.float32)

        lhs_game_index = random.randrange(len(files[first_index]))
        rhs_game_index = random.randrange(len(files[second_index]))
        
        yield (files[first_index][lhs_game_index], files[second_index][rhs_game_index]), won


def train_model(model):
    won_games = []
    lost_games = []
    for w_name in ["white_won", "black_won"]:
        prepare_file(w_name, won_games)
    for l_name in ["white_lost", "black_lost"]:
        prepare_file(l_name, lost_games)
    files = [won_games, lost_games]
    # print("Output: {}".format(model(tf.constant([[won_games[0]], [lost_games[0]]]))))

    output_signature = ((tf.TensorSpec(shape=(8, 8, 12), dtype=tf.float32), tf.TensorSpec(shape=(8, 8, 12), dtype=tf.float32)), tf.TensorSpec(shape=(2,), dtype=tf.float32))
    dataset = tf.data.Dataset.from_generator(data_generator, output_signature=output_signature, args=[files]).take(100_000).batch(32).prefetch(tf.data.AUTOTUNE)
    # train_dataset, test_dataset = tf.keras.utils.split_dataset(dataset, left_size=0.9)
    for _ in range(10):
        print("Calling fit")
        model.fit(dataset, epochs=100)
        
        # test_dataset = tf.data.Dataset.from_generator(data_generator, output_signature=output_signature, args=[files]).take(100_000).batch(16)
        # model.evaluate(test_dataset)

        input_spec = (tf.TensorSpec(shape=(None, 8, 8, 12), dtype=tf.float32), tf.TensorSpec(shape=(None, 8, 8, 12), dtype=tf.float32))
        signatures = { "call": model.call.get_concrete_function(input_spec) }
        model.save("model", save_format="tf", signatures=signatures)


if __name__ == "__main__":
    # if len(sys.argv) > 1 and sys.argv[1] == "new":
        # model = generate_ae.generate_model()
        # generate_ae.train_model(model)
        # encoder = model.encoder
    # else:
        # encoder = keras.models.load_model("enc_model")
    # encoder.trainable = False
    
    n_model = NardsModel()
    optimizer = keras.optimizers.Adam(0.0005)
    accuracy_metric = keras.metrics.BinaryAccuracy(threshold=0.5)
    n_model.compile(optimizer, loss="categorical_crossentropy", metrics=[accuracy_metric, "accuracy", "mae", "mse"])
    train_model(n_model)
