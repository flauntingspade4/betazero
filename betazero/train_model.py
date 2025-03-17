import tensorflow as tf
import sys
import pickle
import tf_keras as keras
import os

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def load_model(dir_name):
    return keras.models.load_model(dir_name)


def prepare_games(moves, inputs, p_outputs, v_outputs):
    # inputs, p_outputs, v_outputs = [], [], []
    for move in moves:
        # print("Appending board {}".format(tensor))
        input = tf.constant(move["board"]["data"], name="input_1")
        input = tf.cast(tf.reshape(input, (8, 8, 12)), tf.uint64)
        # print("Now tensor {}".format(input))
        inputs.append(input)
        p_outputs.append(tf.constant(move["moves"]))
        v_outputs.append(tf.constant(move["won"]))


def load_and_train_dir(model, dir_name="./games"):
    inputs, p_outputs, v_outputs = [], [], []
    for (dirpath, _, file_names) in os.walk(dir_name):
        for name in file_names:
            print("Adding file " + os.path.join(dirpath, name))
            with open(os.path.join(dirpath, name), "rb") as f:
                games = pickle.load(f)
                prepare_games(games, inputs, p_outputs, v_outputs)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, {"policy_output": p_outputs, "value_output": v_outputs })).shuffle(5000).batch(5)
    optimizer = keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss={"policy_output": "categorical_crossentropy", "value_output": "kl_divergence"}, metrics={"policy_output": ["mse", "accuracy"], "value_output": ["mse", "accuracy"]})
    model.fit(dataset, epochs=15, shuffle=True)
    
def load_and_train(model, file_name="latest.pickle"):
    inputs, p_outputs, v_outputs = [], [], []
    with open(file_name, "rb") as f:
        games = pickle.load(f)
        prepare_games(games, inputs, p_outputs, v_outputs)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, {"policy_output": p_outputs, "value_output": v_outputs })).shuffle(500).batch(5)
    optimizer = keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(optimizer=optimizer, loss={"policy_output": "categorical_crossentropy", "value_output": "categorical_crossentropy"}, metrics={"policy_output": ["mse", "accuracy"], "value_output": ["mse", "accuracy"]})

    model.fit(dataset, epochs=5, shuffle=True)


if __name__ == "__main__":
    print(1e-4)
    model = load_model("model")

    load_and_train_dir(model)
    
    while True:
        # board_spec = tf.TensorSpec([None, 8, 8, 12], tf.uint64, name="board")
        # print(model.call.concrete_functions[0])
        # board = [[[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(8)] for i in range(8)]]
        # input = tf.constant(board, dtype=tf.uint64)
        # res = model(input)
        # print("Result:\n" + str(res))

        # break
        import time
        import subprocess
        p = subprocess.Popen("cargo run --release", shell=True)
        p.wait()
        time.sleep(1)
        
        # call_concrete = model.call.get_concrete_function(tf.TensorSpec([None, 8, 8, 12], tf.uint64, name="input_1"))
        # model.save("model", save_format="tf", signatures=signatures)
        load_and_train(model)
        signatures = { "call": model.signatures["call"] }
        model.save("model", save_format="tf", signatures=signatures)
        time.sleep(5)
        model = load_model("model")
