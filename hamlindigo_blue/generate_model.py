from os import listdir
from os.path import isfile, join
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
    def __init__(self, encoder):
        super(NardsModel, self).__init__()
        self.encoder = encoder
        self.concat = keras.layers.Concatenate()
        self.flatten = keras.layers.Flatten()
        self.dense_and_dropout = []
        for unit in UNITS:
            self.dense_and_dropout.append(keras.layers.Dense(unit, activation=leaky, kernel_regularizer=keras.regularizers.L2(L2_REGULARIZATION)))
            self.dense_and_dropout.append(keras.layers.Dropout(0.2))
        self.output_layer = keras.layers.Dense(2, activation="softmax")


    @tf.function()
    def call(self, inputs):
        lhs, rhs = inputs
        lhs = self.encoder(lhs)
        rhs = self.encoder(rhs)
        return self.call_encoded([lhs, rhs])


    @tf.function()
    def call_encoded(self, inputs):
        lhs, rhs = inputs
        x = self.concat([lhs, rhs])
        x = self.flatten(x)
        for layer in self.dense_and_dropout:
            x = layer(x)
        return self.output_layer(x)
    
    @tf.function()
    def encode(self, input):
        return self.encoder(input)
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "encoder": keras.saving.serialize_keras_object(self.encoder),
        }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        sublayer_config = config.pop("encoder")
        encoder = keras.saving.deserialize_keras_object(sublayer_config)
        return cls(encoder, **config)
    
def save_model(model, loaded):
    # When a model is loaded from disk it is no longer a
    # python object, but already has signatures
    if loaded:
        model.save("model", save_format="tf", signatures=model.signatures)
    else:
        board_spec = tf.TensorSpec(shape=(None, 8, 8, 14), dtype=tf.float32)
        call_spec = (board_spec, board_spec)
        call_encoded_spec = (tf.TensorSpec(shape=(None, generate_ae.LATENT_DIM), dtype=tf.float32), tf.TensorSpec(shape=(None, generate_ae.LATENT_DIM), dtype=tf.float32))
        signatures = { "call": model.call.get_concrete_function(call_spec), "call_encoded": model.call_encoded.get_concrete_function(call_encoded_spec), "encode": model.encode.get_concrete_function(board_spec) }
        model.save("model", save_format="tf", signatures=signatures)
        # Save the encoder
        model.encoder.save("enc_model", save_format="tf", signatures=model.encoder.signatures)


def prepare_file(file_name, parsed):
    data = pickle.load(open(file_name, "rb"))
    for position in data:
        parsed.append(tf.reshape(tf.constant(position["data"], dtype=tf.float32), (8, 8, 14)))
    print("Done file {}".format(file_name))

def parse_and_prepare_file(file_name):
    parsed = []
    prepare_file(file_name, parsed)
    return parsed

results = [tf.constant([1.0, 0.0], dtype=tf.float32), tf.constant([0.0, 1.0], dtype=tf.float32)]
def data_generator(white_won, black_won):
    while True:
        # Choose random positions from random files
        white_won_pos = random.choice(random.choice(white_won))
        black_won_pos = random.choice(random.choice(black_won))
        # Randomise the order
        if bool(random.getrandbits(1)):
            yield (white_won_pos, black_won_pos), results[0]
        else:
            yield (black_won_pos, white_won_pos), results[1]
            

def sliding_window(data, size):
    n = len(data)
    if n < size:
        return
    while True:
        random.shuffle(data)
        current_files = [parse_and_prepare_file(data[i]) for i in range(size)]
        for i in range(size, n - size):
            yield current_files
            current_files.pop(0)
            current_files.append(parse_and_prepare_file(data[i]))


def train_model(model, loaded):
    white_won_files = [join("white_won", f) for f in listdir("white_won") if isfile(join("white_won", f))]
    black_won_files = [join("black_won", f) for f in listdir("black_won") if isfile(join("black_won", f))]
    white_won_sliding = sliding_window(white_won_files, 4)
    black_won_sliding = sliding_window(black_won_files, 4)

    output_signature = ((tf.TensorSpec(shape=(8, 8, 14), dtype=tf.float32), tf.TensorSpec(shape=(8, 8, 14), dtype=tf.float32)), tf.TensorSpec(shape=(2,), dtype=tf.float32))
    losses = []
    accuracy = []
    for _ in range(4):
        for _ in range(5):
            # Choose the files to train from
            chosen_white = next(white_won_sliding)
            chosen_black = next(black_won_sliding)
            dataset = tf.data.Dataset.from_generator(data_generator, output_signature=output_signature, args=[chosen_white, chosen_black]).batch(32).take(3000).prefetch(tf.data.AUTOTUNE)
            # Train and save the model
            history = model.fit(dataset, epochs=5)
            print(history.history)
            for l in history.history["loss"]:
                losses.append(l)
            for a in history.history["binary_accuracy"]:
                accuracy.append(a)

        save_model(model, loaded)
        
        # Choose the files to test from
        chosen_white = [parse_and_prepare_file("white_won_test.pickle")]
        chosen_black = [parse_and_prepare_file("black_won_test.pickle")]
        test_dataset = tf.data.Dataset.from_generator(data_generator, output_signature=output_signature, args=[chosen_white, chosen_black]).batch(32).take(1000).prefetch(tf.data.AUTOTUNE)

        model.evaluate(test_dataset)
    print(accuracy)
    generate_ae.plot_losses(losses, "Loss")
    generate_ae.plot_losses(accuracy, "Accuracy")
    return model


if __name__ == "__main__":
    accuracy_metric = keras.metrics.BinaryAccuracy(threshold=0.5)
    
    loaded = False
    if loaded:
        n_model = keras.models.load_model("model")
        optimizer = keras.optimizers.Adam(0.0005)
        n_model.compile(optimizer, loss="categorical_crossentropy", metrics=[accuracy_metric, "mae", "mse"])
    else:
        if len(sys.argv) > 1 and sys.argv[1] == "new":
            model = generate_ae.generate_model()
            generate_ae.train_model(model)
            encoder = model.encoder
        else:
            encoder = keras.models.load_model("enc_model")
        optimizer = keras.optimizers.Adam(learning_rate=0.00005)
        encoder.compile(optimizer, loss="binary_crossentropy", metrics=[accuracy_metric, "mse"])
        n_model = NardsModel(encoder)
        optimizer = keras.optimizers.Adam(0.0005)
        n_model.compile(optimizer, loss="categorical_crossentropy", metrics=[accuracy_metric, "mae", "mse"])

    train_model(n_model, loaded)
