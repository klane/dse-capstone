import time
import sys
import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer

from torchts.nn.models.dcrnn import DCRNN
from torchts.utils import data as utils


outdir = sys.argv[1]
config_filename = outdir + "config.yaml"

with open(config_filename) as f:
    config = yaml.load(f)
    graph_pkl_filename = outdir + config['data'].get('graph_pkl_filename')
    sensor_ids, sensor_id_to_ind, adj_mx = utils.load_graph_data(graph_pkl_filename)
    
data_dict = config.get('data')
data_dict['dataset_dir'] = outdir +  data_dict['dataset_dir']
data_dict['graph_pkl_filename'] = outdir +  data_dict['graph_pkl_filename']

data = utils.load_dataset(**data_dict)
scaler = data['scaler']
model_kwargs = config.get('model')

model = DCRNN(adj_mx, scaler, **model_kwargs)


def run():
    # Define trainer
    trainer = Trainer(max_epochs=1, logger=True,gpus = int(sys.argv[2]))

    start = time.time()
    trainer.fit(model, data["train_loader"], data["val_loader"])
    end = time.time() - start
    print("Training time taken %f"%(end-start))

if __name__ == '__main__':
    run()

# Validate
# vals = trainer.validate(model,data["val_loader"])
# print("Validation Results",vals)

# Test
# test = trainer.test(model,data["test_loader"])
# print("Test Results",vals)
