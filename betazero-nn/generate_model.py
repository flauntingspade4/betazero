import tensorflow as tf
import tf_keras as keras

# set WRAPT_DISABLE_EXTENSIONS=true


# tf.compat.v1.disable_eager_execution()


BLOCKS = 2
FILTERS = 128
BATCH_SIZE = 32


class convolutional_block(keras.layers.Layer):
    def __init__(self, relu = True):
        super().__init__()
        self.conv = keras.layers.Conv2D(FILTERS, (3, 3), padding="same", use_bias=False)
        self.batch_norm = keras.layers.BatchNormalization(axis=-1)
        self.use_relu = relu
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
        self.policy_conv = keras.layers.Conv2D(FILTERS, (3, 3), padding="same", use_bias=False, name="policy_conv")
        self.policy_batch_norm = keras.layers.BatchNormalization(axis=-1, name="policy_batch_norm")
        # self.policy_output = keras.layers.Dense(8 * 8 * 64, name="policy_output")
        self.policy_output = keras.layers.Conv2D(64, (3, 3), padding="same", name="policy_output")
        # Value head
        self.value_conv = keras.layers.Conv2D(32, (3, 3), padding="same")
        self.value_flatten = keras.layers.Flatten()
        self.value_conv_to_vector = keras.layers.Dense(128)
        self.value_relu = keras.layers.ReLU()
        # self.value_conv_to_vector = keras.layers.Conv2D(128, (1, 1), padding="same")
        self.value_output = keras.layers.Dense(3, name="value_output")

    # def build(self):

    @tf.function #(input_signature=[tf.TensorSpec(shape=[BATCH_SIZE, 8, 8, 12], dtype=tf.uint64, name="board")])
    def call(self, board, training=False):
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
        v = self.value_relu(v)
        v = self.value_conv_to_vector(v)
        
        return self.policy_output(p), self.value_output(v)
    
    """
    @tf.function #(input_signature=[(tf.TensorSpec(shape=[BATCH_SIZE, 8, 8, 12], dtype=tf.uint64, name="training_boards"), tf.TensorSpec(shape=[BATCH_SIZE, 8, 8, 12], dtype=tf.uint64, name="policy_predictions"))])
    def train_step(self, training_boards, p_values, v_values):
        with tf.GradientTape() as tape:
            print("x in gradienttape: " + str(training_boards))
            p_pred, v_pred = self(training_boards, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            print("p_values : " + str(p_values) + "\np_pred : " + str(p_pred))
            p_loss = self.compute_loss(y=p_values, y_pred=p_pred)
            v_loss = self.compute_loss(y=v_values, y_pred=v_pred)
            # print("y = " + str(y))
            # print("y_pred = " + str(y_pred))
            # loss = self.compute_loss(y=y, y_pred=y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient([p_loss, v_loss], trainable_vars)

        # Update weights
        self.optimizer.apply(gradients, trainable_vars)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            print("Metric issssss : " + str(metric))
            if metric.name == "loss":
                metric.update_state([p_loss, v_loss])
            else:
                metric.update_state([p_values, v_values, p_pred, v_pred])

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    """

bz_model = betazero_model()

# print(board)
bz_model.build([8, 8, 12])
# bz_model.compile(optimizer="adam", loss={"policy_output": "mse", "value_output": "sparse_categorical_accuracy"}, metrics=["accuracy"])
bz_model.compile(optimizer="adam", loss=["mse", "binary_crossentropy"], metrics=["MSE", "accuracy"])

board = [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(8)] for i in range(8)]
input = tf.constant(board, dtype=tf.uint64)
res = bz_model(input)

# print(str(res))

# bz_model.export("model", input_signatures=bz_model.call.get_concrete_function(tf.TensorSpec(shape=[8, 8, 12], dtype=tf.uint64)))
# bz_model.export("model")

# Call signatures
board_spec = tf.TensorSpec([1, 8, 8, 12], tf.uint64, name="board")
call_concrete = bz_model.call.get_concrete_function(board_spec)
# res = call_concrete(input)
# bz_model.summary()


# Training signatures
# training_boards = tf.TensorSpec([BATCH_SIZE, 8, 8, 12], tf.uint64, name="training_boards")
# p_values = tf.TensorSpec([BATCH_SIZE, 8, 8, 64], tf.float32, name="p_values")
# v_values = tf.TensorSpec([BATCH_SIZE, 3], tf.float32, name="v_values")
# training = bz_model.train_step.get_concrete_function(training_boards, p_values, v_values)

# init = tf.variables_initializer(tf.global_variables(), name="init")

signatures = { "call": call_concrete }
# tf.saved_model.save(bz_model, "model", signatures=signatures)
bz_model.save("model", save_format="tf", signatures=signatures)
