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


def prepare_file(file_name, parsed):
    print("Starting preparing")
    data = pickle.load(open(file_name, "rb"))
    print("Doing file with {} entries".format(len(data)))
    for i, position in enumerate(data):
        if i % 50_000 == 0:
            print("Doing {}th entry in file".format(i))
        # print(position)
        # if i > 200_000:
            # break
        parsed.append(tf.reshape(tf.constant(position["data"], dtype=tf.float32), (8, 8, 14)))
    print("Done file")

results = [tf.constant([1.0, 0.0], dtype=tf.float32), tf.constant([0.0, 1.0], dtype=tf.float32)]
def data_generator(white_won, black_won):
    while True:
        won_index = random.randrange(len(white_won))
        lost_index = random.randrange(len(black_won))
        if bool(random.getrandbits(1)):
            yield (white_won[won_index], black_won[lost_index]), results[0]
            # yield (lost[lost_index], won[won_index]), results[1]
        else:
            yield (black_won[lost_index], white_won[won_index]), results[1]
            # yield (won[won_index], lost[lost_index]), results[0]


def train_model(model):
    white_won = []
    black_won = []
    prepare_file("white_won_teams.pickle", white_won)
    prepare_file("black_won_teams.pickle", black_won)
    output_signature = ((tf.TensorSpec(shape=(8, 8, 14), dtype=tf.float32), tf.TensorSpec(shape=(8, 8, 14), dtype=tf.float32)), tf.TensorSpec(shape=(2,), dtype=tf.float32))
    dataset = tf.data.Dataset.from_generator(data_generator, output_signature=output_signature, args=[white_won, black_won]).take(320_000).batch(32).prefetch(tf.data.AUTOTUNE)
    for _ in range(100):
        model.fit(dataset, epochs=10)
        board_spec = tf.TensorSpec(shape=(None, 8, 8, 14), dtype=tf.float32)
        call_spec = (board_spec, board_spec)
        call_encoded_spec = (tf.TensorSpec(shape=(None, generate_ae.LATENT_DIM), dtype=tf.float32), tf.TensorSpec(shape=(None, generate_ae.LATENT_DIM), dtype=tf.float32))
        encode_spec = (board_spec)
        signatures = { "call": model.call.get_concrete_function(call_spec), "call_encoded": model.call_encoded.get_concrete_function(call_encoded_spec), "encode": model.encode.get_concrete_function(encode_spec) }
        model.save("model", save_format="tf", signatures=signatures)

        # test_dataset = tf.data.Dataset.from_generator(data_generator, output_signature=output_signature, args=[won_games, lost_games]).take(10_000).batch(16)
        # model.evaluate(test_dataset)
        # input_spec = (tf.TensorSpec(shape=(None, 8, 8, 12), dtype=tf.float32))
        # signatures = { "call": model.encoder.signatures["call"] }
        # model.encoder.save("enc_model", save_format="tf", signatures=signatures)
    return model


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "new":
        model = generate_ae.generate_model()
        generate_ae.train_model(model)
        encoder = model.encoder
    else:
        encoder = keras.models.load_model("enc_model")
    # encoder.trainable = False
    optimizer = keras.optimizers.Adam(learning_rate=0.00005)
    accuracy_metric = keras.metrics.BinaryAccuracy(threshold=0.5)
    encoder.compile(optimizer, loss="categorical_crossentropy", metrics=[accuracy_metric, "mse"])
    
    n_model = NardsModel(encoder)
    # n_model = keras.models.load_model("model")
    optimizer = keras.optimizers.Adam(0.001)
    n_model.compile(optimizer, loss="categorical_crossentropy", metrics=[accuracy_metric, "mae", "mse"])

    train_model(n_model)
