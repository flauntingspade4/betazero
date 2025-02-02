import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import random

#Define a custom keras model
class custom_model(keras.Model):
    def __init__(self):
        super(custom_model, self).__init__()
        self.dense_1 = keras.layers.Dense(units=2, name="input", activation="softmax")
        self.dense_2 = keras.layers.Dense(units=1, name="output", activation="sigmoid")
        # self.output = tf.keras.layers.Dense(1, activation="softmax")
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 2], dtype=tf.float32, name="input")])
    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return x
    #Train function called from Rust which uses the keras model innate train_step function
    # todo: dropout
    # @tf.function
    # def training(self, train_data):
        # loss = self.train_step(train_data)['loss']
        # return loss

#Create model
model = custom_model()
model.build((1, 2))
model.__call__(tf.ones([1, 2]))

#Compile model
# model.compile(optimizer='SGD', loss="mean_squared_error", metrics = [])
# model.build((1, 2))
# print(str(model(tf.ones([1, 2]))))
# model.summary()
# for l in model.layers:
    # print(str(l) + " " + str(l.trainable))

#Fit on some data using the keras fit method
# x_train = np.array([[random.random() * 10, random.random() * 10] for _ in range(255)])
# y_train = np.array([value[0] + value[1] for value in x_train])

# history = model.fit(x_train, y_train, batch_size=2, epochs=100)
# history = model.fit(training_data_generator, batch_size=2, epochs=100)
# model.fit(x=np.array([[1.0, 0.0], [0.0,0.0], [0.0, 1.0], [1.0, 1.0]]), y=np.array([[1], [0.0], [1.0], [2.0]]), batch_size=1, epochs=100, verbose=0)

#Predict a value using the keras predict function
# print(f"Test-Prediction: {model.predict(np.array([[1.0, 0.0]]))}")

#Saving the model, explictly adding the concrete functions as signatures
signatures = {'call': model.call.get_concrete_function(tf.TensorSpec(shape=[1, 2], name="input"))}
model.export("model", input_signatures=model.call.get_concrete_function(tf.TensorSpec(shape=[1, 2])))
