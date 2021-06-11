import networkx
import numpy as np
import os


class create_dcrnn_data():
    '''
    This class takes a networkx graph with temporal values modeled into them and generates the adjacency matrix
    and adjacency matrix required by the DCRNN model.
    
    Each node is expected to have a series with index as datetime.
    '''
    
    def __init__(self,graph,**kwargs):
        self.graph = graph
        self.outdir = kwargs.get('outdir','')
        self.horizon = kwargs.get('horizon',12)
        
    def _createDataFrame(self):
        frame = list()
        for i in range(g.number_of_nodes()):
            frame.append(g.nodes[i]['values'])
        return pd.concat(frame,axis = 1).sort_index(inplace = False)
    
    def generateAdjacencyMatrix(self):
        return networkx.adjacency_matrix(self.graph)

    def generate_graph_seq2seq_io_data(self,df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None):
        """
        Return:
        # x: (epoch_size, input_length, num_nodes, input_dim)
        # y: (epoch_size, output_length, num_nodes, output_dim)
        """

        num_samples, num_nodes = df.shape
        data = np.expand_dims(df.values, axis=-1)
        data_list = [data]
        
        if add_time_in_day:
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        if add_day_in_week:
            day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
            day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
            data_list.append(day_in_week)

        data = np.concatenate(data_list, axis=-1)
        
        # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
        x, y = [], []
        
        # t is the index of the last observation.
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
        
        for t in range(min_t, max_t):
            x_t = data[t + x_offsets, ...]
            y_t = data[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        return x, y

    
    def generate_train_val_test(self):
        df = self._createDataFrame()
        # Make sure to test what happens at horizon = 0
        x_offsets = np.sort(
            np.concatenate((np.arange(1 + (-1 * self.horizon), 1, 1),))
        )
        
        # Predict the next one hour
        y_offsets = np.sort(np.arange(1, self.horizon + 1, 1))

        # x: (num_samples, input_length, num_nodes, input_dim)
        # y: (num_samples, output_length, num_nodes, output_dim)
        x, y = self.generate_graph_seq2seq_io_data(
            df,
            x_offsets=x_offsets,
            y_offsets=y_offsets,
            add_time_in_day=True,
            add_day_in_week=False,
        )

        print("x shape: ", x.shape, ", y shape: ", y.shape)

        num_samples = x.shape[0]
        num_test = round(num_samples * 0.2)
        num_train = round(num_samples * 0.7)
        num_val = num_samples - num_test - num_train


        x_train, y_train = x[:num_train], y[:num_train]
        
        x_val, y_val = (
            x[num_train: num_train + num_val],
            y[num_train: num_train + num_val],
        )

        x_test, y_test = x[-num_test:], y[-num_test:]

        datasets = {}
        for cat in ["train", "val", "test"]:
            _x, _y = locals()["x_" + cat], locals()["y_" + cat]
            print(cat, "x: ", _x.shape, "y:", _y.shape)
            np.savez_compressed(
                os.path.join(self.outdir, "%s.npz" % cat),
                x=_x,
                y=_y,
                x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
                y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
            )
            datasets["x_" + cat] = _x
            datasets["y_" + cat] = _y

        return datasets
    
        
