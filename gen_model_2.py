import tensorflow as tf
import tf_keras as keras

# class CustomLayer(keras.Layer):
    # def __init__(self):
        # super(CustomLayer, self).__init__()
        # self.

class CustomModule(keras.Model):

  def __init__(self):
    super(CustomModule, self).__init__()
    self.dense = keras.layers.Dense(2)
    self.output_dense = keras.layers.Dense(1)

  @tf.function(input_signature=[tf.TensorSpec([1, 1], tf.float32, name="input")])
  def call(self, x):
    print('Tracing with', x)
    x = self.dense(x)
    return self.output_dense(x)

module = CustomModule()
module.build([1, 1])
# module.compile(optimizer="adam", loss="mse", metrics=["MSE", "accuracy"])


input = [[3.0]]
input = tf.constant(input)
print("Result: " + str(module(input)))
output_data = module.__call__(input)

# input_spec = tf.TensorSpec([1, 1], tf.float32, name="input")
# call_concrete = module.__call__.get_concrete_function(input_spec)

# signatures = { "call": call_concrete }
# tf.saved_model.save(module, "model_2", signatures)

# module.export("model_2.keras")
module.save("model_2", save_format="tf")