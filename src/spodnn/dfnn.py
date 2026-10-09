"""The DFNN: a fully connected network mapping parameters to reduced coordinates.

For sPOD-NN the inputs are the scaled parameters (t, mu) and the outputs the
scaled time amplitudes and shifts; for POD-NN the outputs are POD amplitudes.
Scaling is not part of the network (see spodnn.scaling).

MLP is the network from DL-ROM/LIB/DFNN.py, unchanged. LitMLPModel wraps it as
a Lightning module, so training, prediction and checkpoints (.ckpt) go through
lightning's Trainer, e.g. trainer.predict(model, dataloaders=..., ckpt_path=...).
"""

import numpy as np
import torch.nn
from numpy import vstack
from numpy import sqrt
from pandas import read_csv
import sys
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import ReLU, ELU, LeakyReLU
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn import L1Loss, HuberLoss
from torch.nn.init import xavier_uniform_
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

import torch
from dataclasses import dataclass
import lightning.pytorch as pl


# model definition
class MLP(Module):
    """n_inputs -> 25 -> 50 -> 75 -> 50 -> n_outputs, with ELU, ELU, ELU, LeakyReLU.

    Weights use Xavier-uniform initialisation. The layer names match the paper's
    weights.pt files, so those state dicts load into this class.
    """

    # define model elements
    def __init__(self, n_inputs, n_outputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.input = Linear(n_inputs, 25)
        xavier_uniform_(self.input.weight)
        self.act1 = ELU()
        # first hidden layer
        self.hidden1 = Linear(25, 50)
        xavier_uniform_(self.hidden1.weight)
        self.act2 = ELU()
        # second hidden layer
        self.hidden2 = Linear(50, 75)
        xavier_uniform_(self.hidden2.weight)
        self.act3 = ELU()
        # third hidden layer
        self.hidden3 = Linear(75, 50)
        xavier_uniform_(self.hidden3.weight)
        self.act4 = LeakyReLU()
        # output layer and output
        self.output = Linear(50, n_outputs)
        xavier_uniform_(self.output.weight)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.input(X)
        X = self.act1(X)
        # first hidden layer
        X = self.hidden1(X)
        X = self.act2(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act3(X)
        # third hidden layer
        X = self.hidden3(X)
        X = self.act4(X)
        # output layer and output
        X = self.output(X)
        return X


class LitMLPModel(pl.LightningModule):
    """Lightning wrapper around MLP.

    The network is stored as the attribute `mlp`, so its weights appear in a
    Lightning checkpoint's state_dict as "mlp.input.weight", "mlp.input.bias", ...

    Args:
        n_inputs: Number of input parameters (2 for (t, mu)).
        n_outputs: Number of outputs (e.g. 18: 16 amplitudes + 2 shifts).
    """

    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        # Pass hyperparameters down to the custom MLP class
        self.mlp = MLP(n_inputs=n_inputs, n_outputs=n_outputs)

    def forward(self, x):
        """Evaluate the network on x of shape (batch, n_inputs); returns (batch, n_outputs)."""
        # Delegate the actual forward pass directly to the inner MLP
        return self.mlp(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction for one batch, called by Trainer.predict.

        The batch must be an (inputs, targets) pair; the targets are ignored.
        """
        # Extract features from the batch (ignores labels/targets during inference)
        x, _ = batch 
        
        # Pass features through the forward method
        preds = self(x)
        
        return preds