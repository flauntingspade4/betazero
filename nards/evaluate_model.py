import tensorflow as tf
import tf_keras as keras

from generate_model import prepare_file, data_generator

if __name__ == "__main__":
    model = keras.models.load_model("model")
    won_games = []
    lost_games = []
    prepare_file("wonsmall.pickle", won_games)
    prepare_file("lostsmall.pickle", lost_games)
    output_signature = ((tf.TensorSpec(shape=(8, 8, 12), dtype=tf.float32), tf.TensorSpec(shape=(8, 8, 12), dtype=tf.float32)), tf.TensorSpec(shape=(2,), dtype=tf.float32))
    dataset = tf.data.Dataset.from_generator(data_generator, output_signature=output_signature, args=[won_games, lost_games]).take(320_000).batch(32).prefetch(tf.data.AUTOTUNE)
    for _ in range(20):
        model.evaluate(dataset)
