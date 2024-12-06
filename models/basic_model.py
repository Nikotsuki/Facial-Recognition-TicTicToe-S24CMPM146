from models.model import Model
import keras
from keras import Sequential, layers
#from layers.experimental.preprocessing import Rescaling
from keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model
        model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Rescaling(1./255, input_shape=input_shape),
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            layers.BatchNormalization(),
            layers.Dropout(0.25),

            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            layers.Dense(categories_count, activation="softmax"),
        ]
)
        self.model = model
        model.save('model.keras')
        #model.print_summary()
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
        self.model.compile(
            optimizer = optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )