import tensorflow as tf
import yaml
from keras.api._v2.keras import datasets, layers, models


class Train:
    def __init__(self):
        pass

    def create_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
        return model

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return (x_train, y_train), (x_test, y_test)

    def train_model(self, model, x_train, y_train, x_test, y_test, num_epochs, batch_size, learning_rate):
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_test, y_test))

    def save_model(self, model):
        model.save('model.pkl')
        model.save('model.keras')
        print("Model saved successfully\nExiting...")


def main():
    with open("../config.yml", "r") as config_file:
        config = yaml.safe_load(config_file)

    EPOCH = config["model"]["epoch"]
    BATCH_SIZE = config["model"]["batch_size"]
    LEARNING_RATE = config["model"]["learning_rate"]

    train = Train()
    (x_train, y_train), (x_test, y_test) = train.load_data()
    model = train.create_model()
    train.train_model(model, x_train, y_train, x_test, y_test, EPOCH, BATCH_SIZE, LEARNING_RATE)
    train.save_model(model)


if __name__ == "__main__":
    main()
