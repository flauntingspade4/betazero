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
    data = pickle.load(open(file_name, "rb"))
    print("Doing file with {} entries".format(len(data)))
    for i, position in enumerate(data):
        # if i < 359000:
            # continue
        parsed.append(tf.reshape(tf.constant(position["board"]["data"], dtype=tf.float32), (8, 8, 12)))
    print("Done file")


def data_generator(won, lost):
    while True:
        won_index = random.randrange(len(won))
        lost_index = random.randrange(len(lost))
        if bool(random.getrandbits(1)):
            result = tf.constant([1.0, 0.0], dtype=tf.float32)
            yield (won[won_index], lost[lost_index]), result
        else:
            result = tf.constant([0.0, 1.0], dtype=tf.float32)
            yield (lost[lost_index], won[won_index]), result


def train_model(model):
    won_games = []
    lost_games = []
    prepare_file("wonsmall.pickle", won_games)
    prepare_file("lostsmall.pickle", lost_games)
    output_signature = ((tf.TensorSpec(shape=(8, 8, 12), dtype=tf.float32), tf.TensorSpec(shape=(8, 8, 12), dtype=tf.float32)), tf.TensorSpec(shape=(2,), dtype=tf.float32))
    dataset = tf.data.Dataset.from_generator(data_generator, output_signature=output_signature, args=[won_games, lost_games]).take(320_000).batch(32).prefetch(tf.data.AUTOTUNE)
    for _ in range(100):
        model.fit(dataset, epochs=10)

        call_spec = (tf.TensorSpec(shape=(None, 8, 8, 12), dtype=tf.float32), tf.TensorSpec(shape=(None, 8, 8, 12), dtype=tf.float32))
        call_encoded_spec = (tf.TensorSpec(shape=(None, generate_ae.LATENT_DIM), dtype=tf.float32), tf.TensorSpec(shape=(None, generate_ae.LATENT_DIM), dtype=tf.float32))
        encode_spec = tf.TensorSpec(shape=(None, 8, 8, 12), dtype=tf.float32)
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
    optimizer = keras.optimizers.Adam(0.001)
    n_model.compile(optimizer, loss="categorical_crossentropy", metrics=[accuracy_metric, "mae", "mse"])

    train_model(n_model)
