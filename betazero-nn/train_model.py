import tensorflow as tf
import sys
import json

print("Num GPUs Available: ", len(tf.config.list_physical_devices()))

def load_model(dir_name):
    return tf.saved_model.load(dir_name)


def prepare_games(games):
    result = []
    for game in games:
        for move in game["moves"]:
            result.append((move["board"], (move["moves"], move["won"])))
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = load_model(sys.argv[1])
    else:
        model = load_model("model")
    
    print(model.signatures["call"])
    board = [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(8)] for i in range(8)]
    input = tf.constant(board, dtype=tf.uint64)
    res = model.signatures["call"](input)
    # with open("games.json", "r") as f:
        # games = json.load(f)
        # games = prepare_games(games)
