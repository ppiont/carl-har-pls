import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import ray
from ray import tune
import scipy.io
from sklearn.metrics import mean_squared_error
import  time

import data_loading
import train_models
import models

#Load data
data_dict = scipy.io.loadmat('../data/data.mat')

Xtest = data_dict['Xtest'] + data_dict['Ex_test']
ytest = data_dict['ytest'] + data_dict['ey_test']

X1 = data_dict['X1']
X2 = data_dict['X2']

y1 = data_dict['y1']
y2 = data_dict['y2']

Ex = data_dict['Ex']
ey = data_dict['ey']

It = data_dict['It'][0, 0]


def train_across_cases(config,case='X1'):
    """Trains NN across all test cases.

    Parameters
    ----------
    config: dictionary
        Hyperparameters
    case: string
        Training case 1 or 2

    Returns
    -------
    MSE: float
        Mean val MSE across all error cases
    """

    batch_size = config['batch_size']
    num_neurons = config['num_neurons']
    num_layers = config['num_layers']
    learning_rate = config['learning_rate']
    activation = config['activation']
    regu = config['regu']

    if activation == 'relu':
        activation = nn.ReLU()
    elif activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation == 'leakyrelu':
        activation = nn.LeakyReLU()
    elif activation == 'tanh':
        activation = nn.Tanh()


    num_epochs = 1000
    patience = 100
    loss_function = nn.MSELoss()

    if case == 'X1':
        X = X1
        y = y1
    elif case == 'X2':
        X = X2
        y = y2

    num_train_samples = 60
    num_val_samples = 40

    val_error = []

    for i in range(It):

        # Define train features and targets
        train_features = X[0:num_train_samples] + Ex[0:num_train_samples, :, i]
        train_targets = y[0:num_train_samples] + ey[0:num_train_samples, i:(i+1)]

        # Create dataloader for training data
        train_loader = data_loading.create_data_loader(features=train_features,
                                                       targets=train_targets,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       drop_last=True)

        # Define val features and targets
        val_features = X[-num_val_samples:] + Ex[-num_val_samples:, :, i]
        val_targets = y[-num_val_samples:] + ey[-num_val_samples:, i:(i+1)]

        # Create dataloader for training data
        val_loader = data_loading.create_data_loader(features=val_features,
                                                     targets=val_targets,
                                                     batch_size=num_val_samples,
                                                     shuffle=True,
                                                     drop_last=True)

        # Instantiate NN
        model = models.NeuralNetwork(n_input=X1.shape[1],
                                     num_neurons=num_neurons,
                                     num_layers=num_layers,
                                     activation=activation)

        model.train(mode=True)

        # Multiple restart training
        model_state_dict = train_models.multstart_training(model=model,
                                                           train_data=train_loader,
                                                           val_data=val_loader,
                                                           loss_function=loss_function,
                                                           regu=regu,
                                                           learning_rate=learning_rate,
                                                           num_epochs=num_epochs,
                                                           patience=patience,
                                                           print_progress=False)
        model.load_state_dict(model_state_dict)
        model.eval()

        # Compute val errors
        _, (X_val, Y_val) =next(enumerate(val_loader))
        pred = model(X_val).detach().numpy()
        val_error.append(mean_squared_error(pred,Y_val))

        del model

    return np.mean(val_error)

def train_network(config):
    MSE = train_across_cases(config, case='X1')
    print(MSE)
    tune.report(mean_loss = MSE)

config={"batch_size": tune.choice([4,8,16,32,64,90]),
        "num_neurons": tune.choice([1, 2, 3, 4, 5]),
        "num_layers": tune.choice([1, 2, 3, 4, 5]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "activation": tune.choice(['relu',
                                   'leakyrelu',
                                   'sigmoid',
                                   'tanh']),
        "regu": tune.loguniform(1e-10, 1e-1)
        }


t1 = time.time()
analysis = tune.run(train_network,
                    config=config,
                    num_samples=100,
                    resources_per_trial={"cpu": 17},
                    verbose=1,
                    metric='mean_loss',
                    mode='min')
t2 = time.time()

print(f'Time: {t2-t1}')

print(f"Best config: {analysis.get_best_config(metric='mean_loss', mode='min')}")
print(f"Best loss: {analysis.best_result['mean_loss']}")

