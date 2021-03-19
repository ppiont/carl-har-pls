import numpy as np
import pdb
import torch.optim
import torch
import copy


def weight_reset(m):
  """Resets all weights in an NN"""
  reset_parameters = getattr(m, "reset_parameters", None)
  if callable(reset_parameters):
      m.reset_parameters()

def train_step(model, features, targets, optimizer, loss_function):
    """Computes an optimizer step and updates NN weights.

    Parameters
    ----------
    model : NeuralNetwork instance
    features : torch.Tensor
      Input features.
    targets : torch.Tensor
      Targets to compare with in loss function.
    optimizer : torch.optim optimizer
      Optimizer to compute the step.
    loss_function : torch.nn loss function
      Loss function to compute loss.

    Returns
    -------
    loss : torch.Tensor
      The computed loss value.
    output : torch.Tensor
      The compute NN output.
    """

    model.zero_grad()
    output = model(features)
    loss = loss_function(output, targets)
    loss.backward()
    optimizer.step()

    return loss, output


def train_network(model, train_data, val_data, optimizer, loss_function,
                  num_epochs=300, patience=20, print_progress=True):
    """Trains NN.

    Parameters
    ----------
    model: NeuralNetwork instance
    train_data: torch.utils.data.DataLoader
      Training data.
    val_data : torch.utils.data.DataLoader
      Validation data.
    optimizer : torch.optim optimizer
      Optimizer to compute the step.
    loss_function : torch.nn loss function
      Loss function to compute loss.
    num_epochs : int
      Number of epochs.
    patience : int
      Patience in early stopping
    print_progress: bool
      Determines if intermediate loss values will be printed

    Returns
    -------
    train_loss : list
      Training loss in each epoch
    val_loss : list
      Validation loss in each epoch
    """

    best_loss = 1e8  # Initialize best loss for early stopping

    train_loss = []  # Initialize training loss list
    val_loss = []  # Initialize val loss list

    # Start training
    for epoch in range(num_epochs):
        train_epoch_loss = 0  # Initialize training epoch loss

        # Loop over mini batches
        for bidx, (features, targets) in enumerate(train_data):

            # Training one step
            loss, predictions = train_step(model, features, targets,
                                           optimizer, loss_function)
            train_epoch_loss += loss.detach()

        train_loss.append(train_epoch_loss)  # Save training loss

        # Compute val loss
        val_epoch_loss = 0  # Initialize val epoch loss
        for bidx_, (features, targets) in enumerate(val_data):
            output = model(features)
            val_epoch_loss += loss_function(output, targets)
        val_loss.append(val_epoch_loss.detach())

        # Early stopping
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss.item()  # Save best loss
            best_model = copy.deepcopy(model.state_dict())  # Save weights of best model
            epochs_no_improve = 0  # Set number of epochs with no improvement to 0
        else:
            epochs_no_improve += 1

            # Check early stopping condition
            if epochs_no_improve == patience:
                # print('Early stopping!' )
                model.load_state_dict(best_model)  # Load weights from the best model
                break

        if print_progress:
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_epoch_loss:.4}, ',
                      end='')
                print(f'Val Loss: {val_epoch_loss:.4}')

    return train_loss, val_loss


def multstart_training(model, train_data, val_data, loss_function,
                       regu = 1e-6, learning_rate=1e-3,
                       num_epochs=300, patience=20, print_progress=True,
                       multstart=3):
    """Trains NN with multiple restarts

    Parameters
    ----------
    model: NeuralNetwork instance
    train_data: torch.utils.data.DataLoader
      Training data.
    val_data : torch.utils.data.DataLoader
      Validation data.
    loss_function : torch.nn loss function
      Loss function to compute loss.
    num_epochs : int
      Number of epochs.
    patience : int
      Patience in early stopping
    print_progress: bool
      Determines if intermediate loss values will be printed
    multstart : int
      Number of training restarts

    Returns
    -------
    best_multstart_model : state_dict
      Weights from the best performing NN
    """

    best_multstart_loss = 1e8  # initialize best loss

    # Train NN multstart times
    for i in range(multstart):

        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                     weight_decay=regu)

        # train NN
        train_loss, val_loss = train_network(model=model,
                                             train_data=train_data,
                                             val_data=val_data,
                                             optimizer=optimizer,
                                             loss_function=loss_function,
                                             num_epochs=num_epochs,
                                             patience=patience,
                                             print_progress=print_progress)

        # check if NN performs best
        if np.min(val_loss) < best_multstart_loss:

            # copy state dict of best model
            best_multstart_model = copy.deepcopy(model.state_dict())

            # define new best loss
            best_multstart_loss = np.min(val_loss)

        # Reset weights
        model.apply(weight_reset)

    return best_multstart_model
