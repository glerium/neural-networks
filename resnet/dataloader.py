from tensorflow import data
from keras.utils import image_dataset_from_directory
import pandas as pd

def load_data(filepath='./tiny-imagenet-200'):
    global labels
    labels = pd.read_table(filepath_or_buffer=filepath+'/wnids.txt', header=None)
    labels = labels.to_numpy().tolist()
    labels = [item[0] for item in labels]
    # print(labels)
    label_dict = {labels[i]: i for i in range(len(labels))}

    train_data = image_dataset_from_directory(
        directory=filepath + '/train',
        labels='inferred',
        label_mode='int',
        class_names=labels,
        batch_size=256,
        image_size=(112,112)
    )

    val_labels = pd.read_csv(filepath_or_buffer=filepath+'/val/val_annotations.txt', sep='\t', header=None)
    val_labels.sort_values(val_labels.columns[0], inplace=True)
    val_labels = val_labels[val_labels.columns[1]].tolist()
    val_data = image_dataset_from_directory(
        directory=filepath + '/val',
        labels=[label_dict[i] for i in val_labels],
        label_mode='int',
        batch_size=256,
        image_size=(112,112),
    )
    return (train_data, val_data)
