import h5py
import numpy as np


def read_training_data(file):
    with h5py.File(file, 'r') as hf:
        data_hz = np.array(hf.get('data_hz'))
        data_vt = np.array(hf.get('data_vt'))
        data_rf = np.array(hf.get('data_rf'))
        label = np.array(hf.get('label'))
        #train_data = np.transpose(data_hz, (0, 3, 2, 1))
        #train_label = np.transpose(label, (0, 3, 2, 1))
        return data_hz, data_vt, data_rf, label
