import tensorflow as tf
import tf_keras as keras

# set WRAPT_DISABLE_EXTENSIONS=true


# tf.compat.v1.disable_eager_execution()


BLOCKS = 40
FILTERS = 256


class convolutional_block(keras.layers.Layer):
    def __init__(self, relu = True):
        super().__init__()
        self.conv = keras.layers.Conv2D(FILTERS, (3, 3), padding="same")
        self.batch_norm = keras.layers.BatchNormalization(axis=-1)
        self.use_relu = relu
        if relu:
            self.relu = keras.layers.ReLU()

    @tf.function #(input_signature=[tf.TensorSpec(shape=[1, 8, 8, FILTERS], dtype=tf.float32, name="input")])
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        return self.relu(x) if self.use_relu else x


class residual_block(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv_1 = convolutional_block()
        self.conv_2 = convolutional_block(False)
        self.relu = keras.layers.ReLU()

    @tf.function #(input_signature=[tf.TensorSpec(shape=[1, 8, 8, FILTERS], dtype=tf.float32, name="input")])
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
        # self.policy_output = keras.layers.Dense(8 * 8 * 64, name="policy_output")
        self.policy_output = keras.layers.Conv2D(64, (3, 3), padding="same", name="policy_output")
        self.policy_softmax = keras.layers.Softmax([1, 2, 3])
        # Value head
        self.value_conv = keras.layers.Conv2D(32, (3, 3), padding="same")
        self.value_flatten = keras.layers.Flatten()
        self.value_conv_to_vector = keras.layers.Dense(128, activation="relu")
        # self.value_conv_to_vector = keras.layers.Conv2D(128, (1, 1), padding="same")
        self.value_output = keras.layers.Dense(3, name="value_output")
        self.value_softmax = keras.layers.Softmax()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 8, 8, 12], dtype=tf.uint64, name="input_1")])
    def call(self, board):
        # Transform data to correct shape and type
        x = tf.cast(board, tf.float32)
        while x.ndim < 4:
            x = tf.expand_dims(x, 0)
        print("x after being expanded: " + str(x))
        # The main body of the network
        x = self.input_conv(x)
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
        
        return {"policy_output": self.policy_output(p), "value_output": self.value_softmax(self.value_output(v))}

def generate_model():
    bz_model = betazero_model()

    # print(board)
    # bz_model.build(board_spec)
    bz_model.compile(optimizer="adam", loss={"policy_output": "categorical_crossentropy", "value_output": "categorical_crossentropy"}, metrics={"policy_output": "mse", "value_output": "accuracy"})
    # bz_model.compile(optimizer="adam", loss=["mse", "binary_crossentropy"], metrics=["MSE", "accuracy"])

    board_spec = tf.TensorSpec([None, 8, 8, 12], tf.uint64, name="input_1")
        # call_concrete = model.__call__.get_concrete_function(tf.TensorSpec([None, 8, 8, 12], tf.uint64, name="input_1"))
    
    board = [[[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(8)] for i in range(8)]]
    input = tf.constant(board, dtype=tf.uint64)
    res = bz_model(input)

    print(str(res))

    # bz_model.export("model", input_signatures=bz_model.call.get_concrete_function(tf.TensorSpec(shape=[8, 8, 12], dtype=tf.uint64)))
    # bz_model.export("model")

    # Call signatures
    call_concrete = bz_model.call.get_concrete_function(board_spec)
    # res = call_concrete(input)
    bz_model.summary()

    print("{} trainable variables".format(len(bz_model.trainable_variables)))
    print("{} variables".format(len(bz_model.variables)))


    # Training signatures
    # training_boards = tf.TensorSpec([None, 8, 8, 12], tf.uint64, name="training_boards")
    # p_values = tf.TensorSpec([None, 8, 8, 64], tf.float32, name="p_values")
    # v_values = tf.TensorSpec([None, 3], tf.float32, name="v_values")
    # train_step_concrete = bz_model.train_step.get_concrete_function(training_boards, p_values, v_values)

    # init = tf.variables_initializer(tf.global_variables(), name="init")

    signatures = { "call": call_concrete } #, "train": train_step_concrete }
    print(signatures)
    bz_model.save("model", save_format="tf", signatures=signatures)
    return bz_model

if __name__ == "__main__":
    generate_model()
