import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

def main():
    model = keras.models.load_model(os.path.join(MODEL_DIR, "model.h5"))
    x_test = np.load(os.path.join(PROCESSED_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))

    metrics = {"test_loss": float(loss), "test_accuracy": float(accuracy)}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Test accuracy: {accuracy:.4f} | Test loss: {loss:.4f}")

if __name__ == "__main__":
    main()