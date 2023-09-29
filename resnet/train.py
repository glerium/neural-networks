from resnet import ResNet
from dataloader import load_data
from keras.optimizers import RMSprop
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

train_data, val_data = load_data()

for data in train_data:
    print(f'Shape of train_data: {data[0].shape}')
    break
for data in val_data:
    print(f'Shape of val_data: {data[0].shape}')
    break

class_num = 200

net = ResNet(output_size=class_num)
loss = SparseCategoricalCrossentropy(from_logits=False)
optimizer = RMSprop(learning_rate=0.001, weight_decay=0.0001, momentum=0.9)

net.compile(optimizer=optimizer, 
            loss=loss, 
            metrics=['accuracy'])

history = net.fit(train_data,
                  epochs=10,
                  batch_size=1,
                  validation_data=val_data,
                  shuffle=True)

