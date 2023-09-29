import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, BatchNormalization, Dropout, Conv2D, MaxPool2D, Softmax
from keras.datasets import mnist
from keras.optimizers import RMSprop
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import ModelCheckpoint

print(tf.config.list_physical_devices())

class ConvBlock(Model):
    def __init__(self, conv_count=2, filters=64, kernel_size=(3,3), pool_size=(2,2), padding='same', activation='relu'):
        super().__init__()
        self.conv = list()
        for i in range(conv_count):
            self.conv.append(Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation))
        self.bn = BatchNormalization()
        self.pool = MaxPool2D(pool_size=pool_size, padding=padding)
        self.dropout = Dropout(rate=0.2)
    
    def call(self, x):
        for layer in self.conv:
            x = layer(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x

class VGGNet(Model):
    def __init__(self):
        super().__init__()
        self.blocks = Sequential([
            ConvBlock(conv_count=2, filters=64),
            ConvBlock(conv_count=2, filters=128),
            ConvBlock(conv_count=4, filters=256),
            ConvBlock(conv_count=4, filters=512),
            ConvBlock(conv_count=4, filters=512),
            Flatten(),
            Dense(512),
            Dense(512),
            Dense(10, activation='softmax'),
        ])
    
    def call(self, x):
        return self.blocks(x)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(float)
x_train /= 255
x_test = x_test.astype(float)
x_test /= 255

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

batch_size = 128
epochs = 10

net = VGGNet()
optimizer = RMSprop()
loss = SparseCategoricalCrossentropy(from_logits=True)
cpcallback = ModelCheckpoint(
    filepath='models/cp-{epoch}.ckpt',
    verbose=1,
    save_weights_only=True, 
    save_freq=batch_size
)

net.compile(optimizer=optimizer, 
            loss=loss,
            metrics=['accuracy'])

net(x_train[0].reshape(-1,28,28,1))

history = net.fit(
    x=x_train, 
    y=y_train, 
    batch_size=batch_size, 
    epochs=50, 
    validation_data=(x_test, y_test), 
    callbacks=[cpcallback]
)

