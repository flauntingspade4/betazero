import tensorflow as tf
import sys
import pickle
import tf_keras as keras

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def load_model(dir_name):
    return keras.models.load_model(dir_name)


def prepare_games(moves):
    inputs, p_outputs, v_outputs = [], [], []
    for move in moves:
        # print("Appending board {}".format(tensor))
        input = tf.constant(move["board"]["data"])
        input = tf.reshape(input, (8, 8, 12))
        # print("Now tensor {}".format(tensor))
        inputs.append(input)
        p_outputs.append(tf.constant(move["moves"]))
        v_outputs.append(tf.constant(move["won"]))
    return tf.cast(inputs, tf.dtypes.uint64), p_outputs, v_outputs


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = load_model(sys.argv[1])
    else:
        model = load_model("model")

    # print(model.signatures["call"])
    # board = [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(8)] for i in range(8)]
    # input = tf.constant(board, dtype=tf.uint64)
    # res = model.signatures["call"](input)
    with open("games2.json", "rb") as f:
        games = pickle.load(f)
        inputs, p_outputs, v_outputs = prepare_games(games)
        dataset = tf.data.Dataset.from_tensor_slices((inputs, {"policy_output": p_outputs, "value_output": v_outputs }))
    data = dataset.batch(50)
    print(data)
    # print(p_outputs[0])
    # print(v_outputs[0])
    model.fit(data)
    # model.fit(inputs, { "policy_output": p_outputs, "value_output": v_outputs })
    # model.train_step(inputs, p_outputs, v_outputs)
    # model.train_step()
    # print("Network has {} trainable variables".format(len(model.trainable_variables)))
    # for i, v in enumerate(model.variables):
        # print("Variable {} is {}".format(i, v.name))
    # print("Network has {} variables".format(len(model.variables)))
    # model.fit(inputs, outputs)
    # print("Network has {} trainable variables".format(len(model.)))