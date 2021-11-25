# This files contains functions that are important to devellop our model and will help us optimise the code
# and the training time.

import tensorflow as tf
from tensorflow import keras

DESIRED_ACCURACY = 0.9899


# this function will stop the training if a certain accuracy is reached
# coded by : Mohamed El Baha
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > DESIRED_ACCURACY:
            print("\nReached 99.9% accuracy, so cancelling training!")
            self.model.stop_training = True
