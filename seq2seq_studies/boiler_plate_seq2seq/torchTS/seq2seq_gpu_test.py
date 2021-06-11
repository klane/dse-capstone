from torchts.utils import data as utils
from torchts.nn.models.seq2seq import Encoder, Decoder, Seq2Seq 

import time
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

dataset_dir = './'

# learning_rate = 0.01
dropout_rate = 0.8
num_layers = 1
hidden_dim = 64
input_dim = 1
output_dim = 1
horizon = 12
batch_size = 64

data = utils.load_dataset(dataset_dir, batch_size=batch_size, test_batch_size=batch_size)

encoder = Encoder(input_dim, hidden_dim, num_layers, dropout_rate)
decoder = Decoder(output_dim, hidden_dim, num_layers, dropout_rate)
model = Seq2Seq(encoder, decoder, output_dim, horizon)

tb_logger = pl_loggers.TensorBoardLogger('logs/')
trainer = Trainer(max_epochs=1, logger=tb_logger, gpus = 0)

start = time.time()
trainer.fit(model, data["train_loader"], data["val_loader"])
print("Training time taken %f"%(time.time() - start), flush=True)

# Validate
vals = trainer.validate(model,data["val_loader"])
print("Validation Results", vals)

# Test
test = trainer.test(model,data["test_loader"])
print("Test Results", vals)

    
    
    
    
    