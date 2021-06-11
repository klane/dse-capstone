import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils import data
import os
from os.path import join, abspath, exists
import time
from torch.autograd import Variable
from torch.utils import data
from datetime import datetime
import pickle as pkl
from collections import defaultdict
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


################################################################# seq2seq model #################################################################

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


def train_model(model, X, Y, learning_rate, output_steps, batch_size, train_idx, valid_idx, test_idx, dropout_rate, hidden_dim, test=False, return_pred=False):
    
    model_path = '/models-vol2/trained_model_{}_{}_{}.pth'.format(learning_rate, dropout_rate, hidden_dim)
    results_path = '/models-vol2/results_dict_{}_{}_{}.pkl'.format(learning_rate, dropout_rate, hidden_dim)


    ############ read checkpoint ############
    
    if exists(model_path):
        model_obj = torch.load(model_path)
        model.load_state_dict(model_obj.state_dict())
        print(datetime.now(), ' Model Loaded {}...'.format(model_path), flush=True)
        write_log('Model Loaded {}...'.format(model_path))

    if exists(results_path):
        with open(results_path, "rb") as fout:
            results_dict = pkl.load(fout)
        fout.close()
        print(datetime.now(), ' Results Loaded {}...'.format(results_path), flush=True)
        write_log('Results Loaded {}...'.format(results_path))
    else:
        results_dict = defaultdict(list)

    ############ read checkpoint ############

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
    

    for i in range(200):
        
        print(datetime.now(), ' Epoch --> {}'.format(i), flush=True)
        write_log('Epoch --> {}'.format(i))

        start = time.time()
        scheduler.step()
        train_generator = data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
        valid_generator = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False)
        
        if test:
            test_generator = data.DataLoader(test_set, batch_size = batch_size, shuffle = False)
 
        model.train()
        train_rmse_value = run_epoch_train(model, train_generator, optimizer, criterion)
        train_rmse.append(train_rmse_value)
            
        model.eval()
        rmse, predictions = run_epoch_eval(model,  valid_generator, criterion, return_pred = True)
        valid_rmse.append(rmse)
        
        if test:
            if return_pred:
                t_rmse, test_predictions = run_epoch_eval(model, test_generator, criterion, return_pred = True)
            else:
                t_rmse = run_epoch_eval(model, test_generator, criterion, return_pred = False)
            test_rmse.append(t_rmse)
        

        ############ save checkpoint ############
        
        torch.save(model, model_path)
        print(datetime.now(), ' Model Saved {}...'.format(model_path), flush=True)
        write_log('Model Saved {}...'.format(model_path))

        results_dict['train_rmse'].append(train_rmse_value)
        results_dict['val_rmse'].append(rmse)
        results_dict['test_rmse'].append(t_rmse)
        
        with open(results_path, 'wb') as handle:
            pkl.dump(results_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
        print(datetime.now(), ' Results Saved {}...'.format(results_path), flush=True)
        write_log('Results Saved {}...'.format(results_path))
        write_log(str(results_dict))
        print(datetime.now(), ' ' + str(results_dict), flush=True)

        ############ save checkpoint ############


        if valid_rmse[-1] < min_loss:
            min_loss = valid_rmse[-1]
            best_model = model
            min_valid_loss = valid_rmse[-1]
            best_preds = predictions
            min_valid_loss = valid_rmse[-1]
            
        if (len(train_rmse) > 15 and np.mean(valid_rmse[-5:]) >= np.mean(valid_rmse[-10:-5])):
            break
            
        end = time.time()       
        print(datetime.now(), (" Epoch %d:"%(i+1)), ("Loss: %f; "%train_rmse[-1]),("valid_loss: %f; "%valid_rmse[-1]), 
              ("Time per epoch: %f; "%round(end - start,5)), flush=True)
        write_log('Epoch {}, Loss {}, valid loss {}, time per epoch {}'.format(i, train_rmse[-1], valid_rmse[-1], round(end - start,5)))

    if test:
        if return_pred:
            return best_model, (train_rmse,valid_rmse),  best_preds, min_valid_loss, test_rmse, test_predictions
        return best_model, (train_rmse,valid_rmse),  best_preds, min_valid_loss, test_rmse
    return best_model, (train_rmse,valid_rmse),  best_preds, min_valid_loss


################################################################# Helper Fns #################################################################


def write_log(msg):

    with open('/models-vol2/training_logger.log', "a+") as f:
        f.write("\n" + datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " " + msg)
    f.close()   


################################################################# Main #################################################################

print(device, flush=True)
write_log(str(device))

print(datetime.now(), ' Code Started...', flush=True)
write_log('#'*100)
write_log('Code Started...')

base_dir = '/models-vol2/'
# filename = 'traffic_bayArea_station_allStations_12pts.pkl'
# filename = 'traffic_bayArea_station_400001.pkl'
filename = 'traffic_bayArea_station_allStations_12pts_SPEED.pkl'

# read tensor
with open(base_dir + filename, "rb") as fout:
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
write_log('sample size {}, train val size {}, train size {}, val size {}, test size {}'.format( sample_size,
                                                                                                train_valid_size,
                                                                                                training_size,
                                                                                                validation_size,
                                                                                                test_size))

X_all = c_time_series[:train_valid_size+test_size,:pred_size,:]
Y_all = c_time_series[:train_valid_size+test_size,pred_size:,:]
X, Y, (avg, std) = scale_data(X_all, Y_all, out_pos = 0, return_current_avg_std = True)



learning_rate = [0.001, 0.0001, 0.0001, 0.001, 0.01]
dropout_rate = [0.8, 0.2, 0.4, 0.6, 0.6]
hidden_dim = [448, 64, 384, 384, 256]

num_layers = 1


input_steps = segment_size
output_steps = segment_size
input_size = 1
output_size = 1

train_idx = list(range(training_size))
valid_idx = list(range(training_size, train_valid_size))
test_idx = list(range(train_valid_size, train_valid_size + test_size))


for i in range(len(learning_rate)):

    print(datetime.now(), ' Hyperparameters: learning rate {}, dropout rate {}, hidden dim {}'.format(learning_rate[i], 
                                                                                                      dropout_rate[i],
                                                                                                      hidden_dim[i]), flush=True)
    write_log('Hyperparameters: learning rate {}, dropout rate {}, hidden dim {}'.format(learning_rate[i], 
                                                                                         dropout_rate[i],
                                                                                         hidden_dim[i]))
    
    encoder = Encoder(input_size, hidden_dim[i], num_layers, dropout_rate[i])
    decoder = Decoder(output_size, hidden_dim[i], num_layers, dropout_rate[i])
    model = Seq2Seq(encoder, decoder, device).to(device)

    model, loss, preds, min_valid_loss, test_rmse = train_model(
        model, X, Y, learning_rate[i], output_steps = output_steps, batch_size = 512,
        train_idx = train_idx, valid_idx = valid_idx, test_idx = test_idx, test=True,
        dropout_rate = dropout_rate[i], hidden_dim = hidden_dim[i])


    print(datetime.now(), ' Completed training current hyperparameters...', flush=True)
    write_log('Completed training current hyperparameters...')

print(datetime.now(), ' Code Ended...', flush=True)
write_log('Code Ended...')
write_log('#'*100)
