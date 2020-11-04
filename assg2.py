import sys
sys.path.append('src/')

from data_loader import DataLoader
from mnist import load_mnist
from model import Model
from multilayer_perceptron import MultilayerPerceptron as MLP, Trainer
from random_fns import Normal, RandomFunction
from supervisor import Supervisor

import gzip
import matplotlib.pyplot as plt
import numpy as np
import pickle


train_data, val_data, _ = load_mnist('data/mnist.pkl.gz')

train_dataloader = DataLoader(train_data, batch_size=128)
val_dataloader = DataLoader(val_data)

nn = MLP(
    layer_sizes=[784, 64, 10],
    random_fn=Normal(std=1)
)

trainer = Trainer(
    learning_rate=0.01,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader
)
supervisor = Supervisor(epsilon=0.1)
epochs = supervisor.run(nn, trainer)
