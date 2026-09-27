import os
import numpy as np
import pandas as pd
import yaml
from tensorflow import keras

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)["train"]

def main():
    params = load_params()
    os.makedirs(MODEL_DIR, exist_ok=True)

    x_train = np.load(os.path.join(PROCESSED_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    x_val = np.load(os.path.join(PROCESSED_DIR, "x_val.npy"))
    y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(params["dense_units"], activation="relu"),
        keras.layers.Dropout(params["dropout_rate"]),
        keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
    )

    model.save(os.path.join(MODEL_DIR, "model.h5"))
    pd.DataFrame(history.history).to_csv(os.path.join(MODEL_DIR, "history.csv"), index=False)
    print(f"Model saved to {MODEL_DIR}/model.h5")

if __name__ == "__main__":
    main()