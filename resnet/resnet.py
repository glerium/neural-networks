import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import (Dense, MaxPool2D, AvgPool2D, Conv2D, BatchNormalization, Flatten, Softmax, ReLU,
                          RandomFlip, RandomRotation, RandomContrast)

data_augmentation = Sequential([
    RandomFlip('horizontal_and_vertical'),
    RandomRotation(0.2)
])

class ResBlock(Model):
    def __init__(self, block_count=2, filters=64, strides=1, x_strides=1):
        super().__init__()

        self.block_count = block_count

        if isinstance(filters, int):
            filters = [filters] * block_count
        elif isinstance(filters, tuple) or isinstance(filters, list):
            if len(filters) != block_count:
                raise ValueError('len(filters) not equals block_count.')
        else:
            raise TypeError('Unsupported value type for filters.')

        if isinstance(strides, int):
            strides = [strides] * block_count
        elif isinstance(strides, tuple) or isinstance(strides, list):
            if len(strides) != block_count:
                raise ValueError('len(strides) not equals block_count.')
        else:
            raise TypeError('Unsupported value type for strides.')

        self.convs = list()
        for i in range(block_count):
            self.convs.append(Conv2D(filters=filters[i], 
                                     kernel_size=3, 
                                     strides=strides[i],
                                     padding='same'))
        self.activations = list()
        for i in range(block_count):
            self.activations.append(ReLU())
        self.bn = list()
        for i in range(block_count):
            self.bn.append(BatchNormalization())

        self.conv_x = Conv2D(filters=filters[-1],
                             kernel_size=1,
                             strides=x_strides,
                             padding='same')
    
    def call(self, x):
        y = x
        for layer in range(self.block_count):
            y = self.convs[layer](y)
            y = self.activations[layer](y)
            y = self.bn[layer](y)
        y += self.conv_x(x)
        return y

class ConvBlock(Model):
    def __init__(self, resblock_count=3, filters=64, first_layer=False):
        super().__init__()
        self.blocks = list()
        for i in range(resblock_count):
            if not first_layer and i == 0:
                self.blocks.append(ResBlock(filters=filters, strides=(2,1), x_strides=2))
            else:
                self.blocks.append(ResBlock(filters=filters, strides=1))
    
    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ResNet18(Model):
    def __init__(self, output_size=1000):
        super().__init__()
        self.data_augmentation = data_augmentation
        self.conv1 = Conv2D(filters=64, kernel_size=7, strides=2, padding='same')
        self.pool1 = MaxPool2D(pool_size=3, strides=2, padding='same')
        self.conv2_x = ConvBlock(resblock_count=2, filters=64, first_layer=True)
        self.conv3_x = ConvBlock(resblock_count=2, filters=128)
        self.conv4_x = ConvBlock(resblock_count=2, filters=256)
        self.conv5_x = ConvBlock(resblock_count=2, filters=512)
        self.pool2 = AvgPool2D()
        self.flatten = Flatten()
        self.dense = Dense(output_size)
        self.softmax = Softmax()

    def call(self, x):
        x = self.data_augmentation(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x
    
class ResNet34(Model):
    def __init__(self, output_size=1000):
        super().__init__()
        self.data_augmentation = data_augmentation
        self.conv1 = Conv2D(filters=64, kernel_size=7, strides=2, padding='same')
        self.pool1 = MaxPool2D(pool_size=3, strides=2, padding='same')
        self.conv2_x = ConvBlock(resblock_count=3, filters=64, first_layer=True)
        self.conv3_x = ConvBlock(resblock_count=4, filters=128)
        self.conv4_x = ConvBlock(resblock_count=6, filters=256)
        self.conv5_x = ConvBlock(resblock_count=3, filters=512)
        self.pool2 = AvgPool2D()
        self.flatten = Flatten()
        self.dense = Dense(output_size)
        self.softmax = Softmax()

    def call(self, x):
        x = self.data_augmentation(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x