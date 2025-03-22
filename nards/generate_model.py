import tensorflow as tf
import tf_keras as keras
import pickle
import generate_ae
import sys

FILTERS = [64, 128, 64, 32]

class NardsModel(keras.Model):
    def __init__(self, encoder):
        super(NardsModel, self).__init__()
        self.encoder = encoder
        self.conv_layers = [keras.layers.Conv2D(x, (3, 3), padding="same", activation="relu") for x in FILTERS]
        self.flatten = keras.layers.Flatten()
        self.output = keras.layers.Dense(3, activation="sigmoid")
    
    @tf.function()
    def call(self, x):
        x = self.encoder(x)
        # Reshape from the latent space
        x = tf.reshape(x, (-1, 8, 8, generate_ae.LAYERS[-1]))
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        return self.output(x)
    

def prepare_moves(moves):
    for move in moves:
        input = tf.reshape(tf.constant(move["board"]["data"]), (8, 8, 12))
        output = tf.constant(move["won"])
        yield input, output
        

def train_model(model):
    with open("latest.pickle", "rb") as f:
        moves = pickle.load(f)
        generator = lambda: prepare_moves(moves)
    output_signature = (tf.TensorSpec(shape=(8, 8, 12), dtype=tf.float32), tf.TensorSpec(shape=(3,), dtype=tf.float32))
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature).shuffle(100000).batch(16)
    train_dataset, test_dataset = tf.keras.utils.split_dataset(dataset, left_size=0.9)
    model.fit(train_dataset, epochs=5)
    
    model.evaluate(test_dataset)
    
    input_spec = tf.TensorSpec(shape=(None, 8, 8, 12), dtype=tf.float32)
    signatures = { "call": model.call.get_concrete_function(input_spec) }
    model.save("ae_model", save_format="tf", signatures=signatures)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "new":
        model = generate_ae.generate_model()
        generate_ae.train_model(model)
        encoder = model.encoder
    else:
        encoder = keras.models.load_model("enc_model")
    
    
    n_model = NardsModel(encoder)
    train_model(n_model)