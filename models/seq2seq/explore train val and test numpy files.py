from os.path import join
import numpy as np
import os

dir_ = r'C:\Users\rmartinez4\Box\Personal Git\Nautilus-seq2seq\data-scalability-spatiotemporal\50'

cat_data = np.load(os.path.join(dir_, "train.npz"))
print(cat_data.files)
print(cat_data['x'].shape)
print(cat_data['y'].shape)
print(cat_data['x_offsets'].shape)
print(cat_data['y_offsets'].shape)
print(' ')

cat_data = np.load(os.path.join(dir_, "val.npz"))
print(cat_data.files)
print(cat_data['x'].shape)
print(cat_data['y'].shape)
print(cat_data['x_offsets'].shape)
print(cat_data['y_offsets'].shape)
print(' ')

cat_data = np.load(os.path.join(dir_, "test.npz"))
print(cat_data.files)
print(cat_data['x'].shape)
print(cat_data['y'].shape)
print(cat_data['x_offsets'].shape)
print(cat_data['y_offsets'].shape)