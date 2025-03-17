import tensorflow as tf
import tf_keras as keras
import time

model = keras.models.load_model("model")

board = [[[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(8)] for i in range(8)]]
input = tf.constant(board, dtype=tf.uint64)
results = []
start = time.time()
for i in range(20):
    results.append(model(input))
    time.sleep(1)
end = time.time()
print(end - start)

board = [[[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(8)] for i in range(8)] for _ in range(20)]
input = tf.constant(board, dtype=tf.uint64)

start = time.time()
res = model(input)
end = time.time()
print(end - start)