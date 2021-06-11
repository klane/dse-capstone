# from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils import data
import os
from os import listdir
from os.path import join, abspath
# import itertools
# import re
# import random
import time
from torch.autograd import Variable
# from scipy.signal import tukey
from torch.utils import data
# from tqdm.notebook import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# import warnings
# warnings.filterwarnings("ignore")

print(device, flush=True)


# import json
from datetime import datetime
# import sqlalchemy as sal
# import getpass
# from urllib.request import urlopen
import pickle as pkl

# import boto3
# import io

# from dotenv import load_dotenv
# load_dotenv(verbose=True)


############# Seq2seq functions

def scale_data(data_x, data_y, out_pos = 0, return_current_avg_std = False):
    """ 
    Arg:
        data_x: features
        data_y: labels
        out_pos: the position of feature of which average and stand deviation will be returned.
    returns:
        1. Normalized features and labels
        2. Average and standard deviation of the selected feature.
    """
    avg = data_x[:,:,out_pos].mean()
    std = data_x[:,:,out_pos].std()
#     c_avg = data_x[:,:,1].mean()
#     c_std = data_x[:,:,1].std()
    for i in range(data_x.shape[-1]):
        data_x[:,:,i] = (data_x[:,:,i] - data_x[:,:,i].mean())/data_x[:,:,i].std()
    data_y = (data_y-avg)/std
    if return_current_avg_std:
        return data_x, data_y, (avg, std)  
#         return data_x, data_y, (avg, std), (c_avg, c_std)   
    else:
        return data_x, data_y, (avg, std)

class Dataset(data.Dataset):
    def __init__(self, X, Y, lst_index, output_steps, position_embedding = (False)):
        """
        Args:
            lst_index: indexes of observations in the dataset.
            output_steps: Forecasting Horizon.
        """
        self.X = X[lst_index]
        self.Y = Y[lst_index]
        self.output_steps = output_steps
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index][:self.output_steps]
        return x, y
        

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        """
        Args:
            input_dim: the dimension of input sequences.
            hidden_dim: number hidden units.
            num_layers: number of encode layers.
            dropout_rate: recurrent dropout rate.
        """
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            bidirectional = True, dropout = dropout_rate, batch_first = True)
        
    def forward(self, source):
        """
        Args:
            source: input tensor(batch_size*input dimension)
        Return:
            outputs: Prediction
            concat_hidden: hidden states
        """
        outputs, hidden = self.lstm(source)
        return outputs, hidden
    
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, dropout_rate):
        """
        Args:
            output_dim: the dimension of output sequences.
            hidden_dim: number hidden units.
            num_layers: number of code layers.
            dropout_rate: recurrent dropout rate.
        """
        super(Decoder, self).__init__()
        
        # Since the encoder is bidirectional, decoder has double hidden size
        self.lstm = nn.LSTM(output_dim, hidden_dim*2, num_layers = num_layers, 
                            dropout = dropout_rate, batch_first = True)
        
        self.out = nn.Linear(hidden_dim*2, output_dim)
      
    def forward(self, x, hidden):
        """
        Args:
            x: prediction from previous prediction.
            hidden: hidden states from previous cell.
        Returns:
            1. prediction for current step.
            2. hidden state pass to next cell.
        """
        output, hidden = self.lstm(x, hidden)   
        prediction = self.out(output.float())
        return prediction, hidden     
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        """
        Args:
            encoder: Encoder object.
            decoder: Decoder object.
            device: 
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target_tensor):
        """
        Args:
            source: input tensor.
            target_length: forecasting steps.
        Returns:
            total prediction
        """
        batch_size = source.size(0) 
        input_length = source.size(1) 
        target_length = target_tensor.shape[1]
        output_dim = target_tensor.shape[-1]
        encoder_hidden = (torch.zeros(self.encoder.num_layers*2, batch_size, self.encoder.hidden_dim, device=device),
                          torch.zeros(self.encoder.num_layers*2, batch_size, self.encoder.hidden_dim, device=device))
        encoder_output, encoder_hidden = self.encoder(source)
        
        # Concatenate the hidden states of both directions.
        num_layers = int(encoder_hidden[0].shape[0]/2)
        h = torch.cat([encoder_hidden[0][0:self.encoder.num_layers,:,:], 
                       encoder_hidden[0][-self.encoder.num_layers:,:,:]], 
                      dim=2, out=None).to(device)
        c = torch.cat([encoder_hidden[1][0:self.encoder.num_layers,:,:], 
                       encoder_hidden[1][-self.encoder.num_layers:,:,:]], 
                      dim=2, out=None).to(device)
        concat_hidden = (h, c)
        
        
        outputs = torch.zeros(batch_size, target_length, output_dim).to(self.device)
        decoder_output = torch.zeros((batch_size, 1, output_dim), device = self.device)
        decoder_hidden = concat_hidden
        
        for t in range(target_length):  
            decoder_output, decoder_hidden = self.decoder(decoder_output, decoder_hidden)
            outputs[:,t:t+1,:] = decoder_output
        return outputs


def run_epoch_train(model, data_generator, model_optimizer, criterion):
    """
    Args:
        model: RNN model.
        data_generator: data.DataLoader object.
        model_optimizer: optimizer.
        criterion: loss function
    Returns:
        Root Mean Square Error on Training Dataset
    """
    MSE = []
    for x, y in data_generator:
        # The input shape for nn.conv1d should sequence_length * batch_size * #features
        input_tensor, target_tensor = x.to(device).float(), y.to(device).float()
        model_optimizer.zero_grad()
        loss = 0
        output = model(input_tensor, target_tensor).reshape(target_tensor.shape)
        num_iter = output.size(0)
        for ot in range(num_iter):
            loss += criterion(output[ot], target_tensor[ot])
        MSE.append(loss.item()/num_iter)
        loss.backward()
        model_optimizer.step()
    
    return round(np.sqrt(np.mean(MSE)), 5)
 

def run_epoch_eval(model, data_generator, criterion, return_pred = False):
    """
    Args:
        model: CNN model.
        data_generator: data.DataLoader object.
        criterion: loss function
    Returns:
        Root Mean Square Error on evaluation datasets.
    """
    with torch.no_grad():
        MSE = []
        preds = []
        for x, y in data_generator:
            input_tensor, target_tensor = x.to(device).float(), y.to(device).float()
            loss = 0
            output = model(input_tensor, target_tensor).reshape(target_tensor.shape)
            preds.append(output.cpu().detach().numpy())
            num_iter = output.size(0)
            
            for ot in range(num_iter):
                loss += criterion(output[ot], target_tensor[ot])
            MSE.append(loss.item()/num_iter)
            
    if return_pred == True:
        preds =  np.concatenate(preds).squeeze(-1)
        return round(np.sqrt(np.mean(MSE)), 5), preds
    else:
        return round(np.sqrt(np.mean(MSE)), 5)


def train_model(model, X, Y, learning_rate, output_steps, batch_size, train_idx, valid_idx, test_idx, test=False, return_pred=False):
    # Initialize the model and define optimizer, learning rate decay and criterion
    optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 5, gamma=0.8)
    criterion = nn.MSELoss()
    
    # Split dataset into training set, validation set and test set.
    train_rmse, train_set = [], Dataset(X, Y, train_idx, output_steps)
    valid_rmse, valid_set = [], Dataset(X, Y, valid_idx, output_steps)
    if test:
        test_rmse, test_set = [], Dataset(X, Y, test_idx, output_steps)
    
    min_loss = 1000
    best_model = 0
    best_preds = 0
    min_valid_loss = 1000
    
    # for i in tqdm(range(200)):
    for i in range(200):
        
        print(i, datetime.now(), flush=True)
        
        start = time.time()
        scheduler.step()
        train_generator = data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
        valid_generator = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False)
        if test:
            test_generator = data.DataLoader(test_set, batch_size = batch_size, shuffle = False)
 
        model.train()
        train_rmse.append(run_epoch_train(model, train_generator, optimizer, criterion))
            
        model.eval()
        rmse, predictions = run_epoch_eval(model,  valid_generator, criterion, return_pred = True)
        valid_rmse.append(rmse)
        
        if test:
            if return_pred:
                t_rmse, test_predictions = run_epoch_eval(model, test_generator, criterion, return_pred = True)
            else:
                t_rmse = run_epoch_eval(model, test_generator, criterion, return_pred = False)
            test_rmse.append(t_rmse)
        
        if valid_rmse[-1] < min_loss:
            min_loss = valid_rmse[-1]
            best_model = model
            min_valid_loss = valid_rmse[-1]
            best_preds = predictions
            min_valid_loss = valid_rmse[-1]
            
        if (len(train_rmse) > 15 and np.mean(valid_rmse[-5:]) >= np.mean(valid_rmse[-10:-5])):
            break
            
    end = time.time()       
    print(("Epoch %d:"%(i+1)), ("Loss: %f; "%train_rmse[-1]),("valid_loss: %f; "%valid_rmse[-1]), 
          ("Time: %f; "%round(end - start,5)), flush=True)

    if test:
        if return_pred:
            return best_model, (train_rmse,valid_rmse),  best_preds, min_valid_loss, test_rmse, test_predictions
        return best_model, (train_rmse,valid_rmse),  best_preds, min_valid_loss, test_rmse
    return best_model, (train_rmse,valid_rmse),  best_preds, min_valid_loss


############# Supporting functions for S3

# def aws_session(region_name='us-west-2'):
#     return boto3.session.Session(aws_access_key_id='ASIAT6C7QCXQZ6R6YKRU',
#                                 aws_secret_access_key='Dbra3CydVtifiqvXtNZDgt+dY3PhzNBJ2I7vr8J1',
#                                 region_name=region_name)


# def make_bucket(name, acl):
#     session = aws_session()
#     s3_resource = session.resource('s3')
#     return s3_resource.create_bucket(Bucket=name, ACL=acl)


# def upload_file_to_bucket(bucket_name, file_path):
#     session = aws_session()
#     s3_resource = session.resource('s3')
#     file_dir, file_name = os.path.split(file_path)

#     bucket = s3_resource.Bucket(bucket_name)
#     bucket.upload_file(
#       Filename=file_path,
#       Key=file_name,
#       ExtraArgs={'ACL': 'public-read'}
#     )

#     s3_url = f"https://{bucket_name}.s3.amazonaws.com/{file_name}"
#     return s3_url


# def download_file_from_bucket(bucket_name, s3_key, dst_path):
#     session = aws_session()
#     s3_resource = session.resource('s3')
#     bucket = s3_resource.Bucket(bucket_name)
#     bucket.download_file(Key=s3_key, Filename=dst_path)
 

# def upload_data_to_bucket(bytes_data, bucket_name, s3_key):
#     session = aws_session()
#     s3_resource = session.resource('s3')
#     obj = s3_resource.Object(bucket_name, s3_key)
#     obj.put(ACL='private', Body=bytes_data)

#     s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
#     return s3_url


# def download_data_from_bucket(bucket_name, s3_key):
#     session = aws_session()
#     s3_resource = session.resource('s3')
#     obj = s3_resource.Object(bucket_name, s3_key)
#     io_stream = io.BytesIO()
#     obj.download_fileobj(io_stream)

#     io_stream.seek(0)
#     data = io_stream.read().decode('utf-8')

#     return data


############# Main

print('Code started', datetime.now(), flush=True)

# base_dir = r'C:\Users\rmartinez4\Box\Personal Git\dse-capstone\seq2seq_example\data_traffic'
base_dir = '/models-vol/'

filename = 'traffic_bayArea_station_mediumSize_12pts.pkl'

# read tensor
with open(base_dir + filename, "rb") as fout:
# with open(join(base_dir, filename), "rb") as fout:
    c_time_series = pkl.load(fout)

sample_size = c_time_series.shape[0]
segment_size = c_time_series.shape[1]
pred_size = int(segment_size/2)

test_size = sample_size // 5
train_valid_size = test_size * 4
training_size = test_size * 7//2
validation_size = test_size * 1//2

print(' sample_size', sample_size, '\n',
      'train_valid_size', train_valid_size, '\n',
      'training_size', training_size, '\n',
      'validation_size', validation_size, '\n',
      'test_size', test_size, flush=True)

# X_train = c_time_series[:train_valid_size,:pred_size,:]
# Y_train = c_time_series[:train_valid_size,pred_size:,:]
# x_train_size, pred_length, feature_count = X_train.shape

# X_test = c_time_series[training_size:training_size+test_size,:pred_size,:]
# Y_test = c_time_series[training_size:training_size+test_size,pred_size:,:]

X_all = c_time_series[:train_valid_size+test_size,:pred_size,:]
Y_all = c_time_series[:train_valid_size+test_size,pred_size:,:]

X, Y, (avg, std) = scale_data(X_all, Y_all, out_pos = 0, return_current_avg_std = True)


learning_rate = 0.01
dropout_rate = 0.6
num_layers = 1
hidden_dim = 128

input_steps = segment_size
output_steps = segment_size
input_size = 1
output_size = 1

train_idx = list(range(training_size))
valid_idx = list(range(training_size, train_valid_size))
test_idx = list(range(train_valid_size, train_valid_size + test_size))

encoder = Encoder(input_size, hidden_dim, num_layers, dropout_rate)
decoder = Decoder(output_size, hidden_dim, num_layers, dropout_rate)
model = Seq2Seq(encoder, decoder, device).to(device)

model, loss, preds, min_valid_loss, test_rmse = train_model(
    model, X, Y, learning_rate, output_steps = output_steps, batch_size = 64,
    train_idx = train_idx, valid_idx = valid_idx, test_idx = test_idx, test=True)

results_dict = {
    'learning_rate': learning_rate,
    'dropout_rate': dropout_rate,
    'num_layers':num_layers,
    'hidden_dim': hidden_dim,
    # 'model_state_dict': model.state_dict(),
    'train_rmse': loss[0],
    'valid_rmse': loss[1],
    'test_rmse': test_rmse,
    'min_valid_loss': min_valid_loss
    # 'preds':preds
}


# read saved model
# model = torch.load(r'C:\Users\rmartinez4\Box\Personal Git\dse-capstone\seq2seq_example\Nautilus\models\traffic_bayArea_station_400001_model.pth', map_location=torch.device('cpu'))

# save results dict and model
file_id = filename.split('.')[0]
with open(base_dir + file_id + '_results_dict.pkl', 'wb') as handle:
# with open(join(base_dir, file_id + '_results_dict.pkl'), 'wb') as handle:
    pkl.dump(results_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

torch.save(model, base_dir + file_id + '_model.pth')
# torch.save(model, join(base_dir, file_id + '_model.pth'))


