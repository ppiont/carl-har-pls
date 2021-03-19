import os
import numpy as np
import pdb
import scipy.io
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

import data_loading
import train_models
import models



#Load data
data_dict = scipy.io.loadmat('../data/data.mat')

Xtest = data_dict['Xtest'] + data_dict['Ex_test']
ytest = data_dict['ytest'] + data_dict['ey_test']
#Ex_test = data_dict['Ex_test']
#ey_test = data_dict['ey_test']

X1 = data_dict['X1']
X2 = data_dict['X2']

y1 = data_dict['y1']
y2 = data_dict['y2']

Ex = data_dict['Ex']
ey = data_dict['ey']

It = data_dict['It'][0, 0]

# Define NN and training specification
batch_size = 8
num_train_samples = 90
num_val_samples = 10
num_neurons = 2
activation = nn.Sigmoid()
learning_rate = 1e-2
regu = 1e-6
loss_function = nn.MSELoss()
patience = 200
num_epochs = 1000
path = 'saved_model_weights'

train_model = True # True: model will be trained, False: weights will be loaded
save_weights = True # Whether you want to save the weights after training

X1_case = True # Train or load ANN for X1 case
X2_case = False # Train or load ANN for X2 case

if X1_case:
    ANN_MSE_X1 = []
    ANN_std_X1 = []
    for num_layers in [1,2,3]:

        layer_score_X1 = []

        for i in range(It):

            # Define train features and targets
            train_features = X1[0:num_train_samples] + Ex[0:num_train_samples, :, i]
            train_targets = y1[0:num_train_samples] + ey[0:num_train_samples, i:(i+1)]

            # Create dataloader for training data
            train_loader = data_loading.create_data_loader(features=train_features,
                                                           targets=train_targets,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           drop_last=True)

            # Define val features and targets
            val_features = X1[-num_val_samples:] + Ex[-num_val_samples:, :, i]
            val_targets = y1[-num_val_samples:] + ey[-num_val_samples:, i:(i+1)]

            # Create dataloader for training data
            val_loader = data_loading.create_data_loader(features=val_features,
                                                         targets=val_targets,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         drop_last=True)

            # Instantiate NN
            model = models.NeuralNetwork(n_input=X1.shape[1],
                                         num_neurons=num_neurons,
                                         num_layers=num_layers,
                                         activation=activation)

            model.train(mode=True)

            # Define path for either loading or saving model weights
            X1_path = path + f'/ANN_X1/layers_{num_layers}/neurons_{num_neurons}/'

            if train_model:

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

                # Save weights
                if save_weights:

                    # If directory exists: save weights
                    if os.path.isdir(X1_path):
                        torch.save(model.state_dict(),
                                   X1_path + f'Ex_{i}')

                    # If directiry does NOT exist: Create directory and save weights
                    else:
                        os.makedirs(X1_path)
                        torch.save(model.state_dict(),
                                   X1_path + f'Ex_{i}')

            # Load weights
            else:
                model.load_state_dict(torch.load(X1_path + f'Ex_{i}'))

            # Compute test errors
            pred = model(torch.Tensor(Xtest)).detach().numpy()
            test_error = mean_squared_error(pred,ytest)

            layer_score_X1.append(test_error)

        # Save mean and std of test errors for all error cases
        ANN_MSE_X1.append(np.mean(layer_score_X1))
        ANN_std_X1.append(np.std(layer_score_X1))

        print(f'X1 case with {num_layers} layers done', end=' ')
        print(f'with mean test MSE: {ANN_MSE_X1[-1]:0.4f}')


if X2_case:
    ANN_MSE_X2 = []
    ANN_std_X2 = []
    for num_layers in [1,2,3]:

        layer_score_X2 = []

        for i in range(It):

            # Define train features and targets
            train_features = X2[0:num_train_samples] + Ex[0:num_train_samples, :, i]
            train_targets = y2[0:num_train_samples] + ey[0:num_train_samples, i:(i+1)]

            # Create dataloader for training data
            train_loader = data_loading.create_data_loader(features=train_features,
                                                           targets=train_targets,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           drop_last=True)

            # Define val features and targets
            val_features = X2[-num_val_samples:] + Ex[-num_val_samples:, :, i]
            val_targets = y2[-num_val_samples:] + ey[-num_val_samples:, i:(i+1)]

            # Create dataloader for training data
            val_loader = data_loading.create_data_loader(features=val_features,
                                                         targets=val_targets,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         drop_last=True)

            # Instantiate NN
            model = models.NeuralNetwork(n_input=X2.shape[1],
                                         num_neurons=num_neurons,
                                         num_layers=num_layers,
                                         activation=activation)
            
            model.train(mode=True)

            # Define path for either loading or saving model weights
            X2_path = path + f'/ANN_X2/layers_{num_layers}/neurons_{num_neurons}/'

            if train_model:

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


                # Save weights
                if save_weights:

                    # If directiry exists: Create directory and save weights
                    if os.path.isdir(X2_path):
                        torch.save(model.state_dict(),
                                   X2_path + f'Ex_{i}')

                    # If directiry does NOT exist: Create directory and save weights
                    else:
                        os.makedirs(X2_path)
                        torch.save(model.state_dict(),
                                   X2_path + f'Ex_{i}')

            # Load weights
            else:
                model.load_state_dict(torch.load(X2_path + f'Ex_{i}'))

            # Compute test errors
            pred = model(torch.Tensor(Xtest)).detach().numpy()
            test_error = mean_squared_error(pred,ytest)

            layer_score_X2.append(test_error)

        # Save mean and std of test errors for all error cases
        ANN_MSE_X2.append(np.mean(layer_score_X2))
        ANN_std_X2.append(np.std(layer_score_X2))

        print(f'X2 case with {num_layers} layers done', end=' ')
        print(f'with mean test MSE: {ANN_MSE_X2[-1]:0.4f}')

# PLS training and testing
PLS_MSE_X1 = []
PLS_MSE_X2 = []
PLS_std_X1 = []
PLS_std_X2 = []
for components in [1, 2, 3]:
    component_score_X1 = []
    component_score_X2 = []
    for i in range(It):
        PLSR_X1 = PLSRegression(n_components=components)
        PLSR_X2 = PLSRegression(n_components=components)
        PLSR_X1.fit(X1 + Ex[:, :, i], y1 + ey[:, i:(i + 1)])
        PLSR_X2.fit(X2 + Ex[:, :, i], y2 + ey[:, i:(i + 1)])

        component_score_X1.append(
            mean_squared_error(PLSR_X1.predict(Xtest), ytest))
        component_score_X2.append(
            mean_squared_error(PLSR_X2.predict(Xtest), ytest))

    PLS_MSE_X1.append(np.mean(component_score_X1))
    PLS_MSE_X2.append(np.mean(component_score_X2))
    PLS_std_X1.append(np.std(component_score_X1))
    PLS_std_X2.append(np.std(component_score_X2))

# Ridge regression training and testing
ridge_MSE_X1 = []
ridge_MSE_X2 = []
ridge_std_X1 = []
ridge_std_X2 = []
for alpha in [0.01]:
  alpha_score_X1 = []
  alpha_score_X2 = []
  for i in range(It):
    ridge_X1 = Ridge(alpha = alpha)
    ridge_X2 = Ridge(alpha = alpha)
    ridge_X1.fit(X1+Ex[:,:,i],y1+ey[:,i:(i+1)])
    ridge_X2.fit(X2+Ex[:,:,i],y2+ey[:,i:(i+1)])

    alpha_score_X1.append(mean_squared_error(ridge_X1.predict(Xtest),ytest))
    alpha_score_X2.append(mean_squared_error(ridge_X2.predict(Xtest),ytest))

  ridge_MSE_X1.append(np.mean(alpha_score_X1))
  ridge_MSE_X2.append(np.mean(alpha_score_X2))
  ridge_std_X1.append(np.std(alpha_score_X1))
  ridge_std_X2.append(np.std(alpha_score_X2))

# PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT
plt.figure()
plt.errorbar([1,2,3],PLS_MSE_X1,yerr=PLS_std_X1,linewidth=3,label='X1 PLS')
plt.errorbar([1,2,3],PLS_MSE_X2,yerr=PLS_std_X2,linewidth=3,label='X2 PLS')
#plt.errorbar([1,2,3],ridge_MSE_X1,yerr=ridge_std_X1,linewidth=3,label='X1 Ridge')
#plt.errorbar([1,2,3],ridge_MSE_X2,yerr=ridge_std_X2,linewidth=3,label='X2 Ridge')
if X1_case:
    plt.errorbar([1,2,3],ANN_MSE_X1,yerr=ANN_std_X1,linewidth=3,label='X1 ANN')
if X2_case:
    plt.errorbar([1,2,3],ANN_MSE_X2,yerr=ANN_std_X2,linewidth=3,label='X2 ANN')
plt.grid()
plt.legend(loc='best')
plt.xlabel('PLS Components / Ridge Regu / ANN Layers')
plt.ylabel('MSE')
plt.yscale('log')
plt.savefig('PLS_ANN_Ridge_MSE')
plt.show()