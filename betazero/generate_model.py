import tensorflow as tf
import tf_keras as keras

# set WRAPT_DISABLE_EXTENSIONS=true


# tf.compat.v1.disable_eager_execution()


BLOCKS = 10
FILTERS = 256


class convolutional_block(keras.layers.Layer):
    def __init__(self, relu = True):
        super().__init__()
        self.conv = keras.layers.Conv2D(FILTERS, (3, 3), padding="same")
        self.batch_norm = keras.layers.BatchNormalization(axis=-1)
        self.use_relu = relu
        if relu:
            self.relu = keras.layers.ReLU()
        self.dropout = keras.layers.Dropout(0.2)

    @tf.function
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = self.relu(x) if self.use_relu else x
        return self.dropout(x)


class residual_block(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv_1 = convolutional_block()
        self.conv_2 = convolutional_block(False)
        self.relu = keras.layers.ReLU()

    @tf.function
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        return self.relu(x + inputs)


class betazero_model(keras.Model):
    def __init__(self):
        super(betazero_model, self).__init__()
        self.input_conv = keras.layers.Conv2D(FILTERS, (3, 3), padding="same", data_format="channels_last")
        self.residual_blocks = [residual_block() for _ in range(BLOCKS)]
        # Policy head
        self.policy_conv = keras.layers.Conv2D(FILTERS, (3, 3), padding="same", name="policy_conv")
        self.policy_batch_norm = keras.layers.BatchNormalization(axis=-1, name="policy_batch_norm")
        self.policy_output = keras.layers.Conv2D(64, (3, 3), padding="same", name="policy_output", activation="softmax")
        # Value head
        self.value_conv = keras.layers.Conv2D(32, (3, 3), padding="same")
        self.value_flatten = keras.layers.Flatten()
        self.value_conv_to_vector = keras.layers.Dense(128, activation="relu")
        self.value_output = keras.layers.Dense(3, name="value_output", activation="softmax")

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 8, 8, 14], dtype=tf.float32, name="input_1")])
    def call(self, board):
        # The main body of the network
        x = self.input_conv(board)
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        # Policy head
        p = self.policy_conv(x)
        p = self.policy_batch_norm(p)
        # Value head
        v = self.value_conv(x)
        v = self.value_flatten(v)
        # v = self.value_relu(v)
        v = self.value_conv_to_vector(v)
        
        return {"policy_output": self.policy_output(p), "value_output": self.value_output(v)}

def generate_model():
    bz_model = betazero_model()

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    bz_model.compile(optimizer=optimizer, loss={"policy_output": "categorical_crossentropy", "value_output": "categorical_crossentropy"}, metrics={"policy_output": ["mse", "accuracy"], "value_output": ["mse", "accuracy"]})

    board_spec = tf.TensorSpec([None, 8, 8, 14], tf.float32, name="input_1")
    
    board = [[[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] for _ in range(8)] for i in range(8)]]
    input = tf.constant(board, dtype=tf.float32)
    _ = bz_model(input)

    # Call signatures
    signatures = { "call": bz_model.call.get_concrete_function(board_spec) }
    bz_model.save("model", save_format="tf", signatures=signatures)
    return bz_model

if __name__ == "__main__":
    generate_model()
