"""
Model Training Script for Mesh Deformation Prediction
-----------------------------------------------------
This script trains a neural network to predict deformation deltas for mesh vertices
based on their position and applied inflation factor.

Responsibilities:
- Loads collected data.
- Trains a neural network with batch normalization and dropout.
- Implements early stopping and learning rate scheduling.
- Saves the trained model in ONNX format for deployment.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import os
import numpy as np
import random

def load_data(data_path):
    """
    Load and prepare training data for training.
    """
    X_train, y_train = torch.load(data_path)
    # onnx deformer expects float32
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    return X_train_tensor, y_train_tensor

class DeltaPredictionModel(nn.Module):
    """
    A neural network model for predicting the deltas of mesh points based on their positions
    and the inflation factor.

    Args:
        dropout_rate (float): The dropout rate to apply to each dropout layer (default is 0.2).
    """
    def __init__(self, input_size, dropout_rate=0.2):
        super(DeltaPredictionModel, self).__init__()
        # Input layer
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Hidden layers with residual connections
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.residual_fc2 = nn.Linear(256, 128)
        self.residual_fc3 = nn.Linear(128, 64)

        # Output layer
        self.fc_out = nn.Linear(64, 3)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (Tensor): Input tensor containing the point positions and inflation factor (batch_size, 7).

        Returns:
            Tensor: The predicted deltas (delta_x, delta_y, delta_z) for each point in the batch.
        """
        # Ensure float32 type
        x = x.to(torch.float32)

        # Extract inflation factor separately
        inflation_factor = x[:, -1].unsqueeze(1)  # Shape: (batch_size, 1)

        # First fully connected layer
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x1 = torch.relu(x1)
        x1 = self.dropout1(x1)

        # Residual block 1
        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = torch.relu(x2)
        x2 = self.dropout2(x2)

        # Add a **residual connection** (ensure dimensions match)
        x2 = x2 + self.residual_fc2(x1)  # Project x1 if needed

        # Residual block 2
        x3 = self.fc3(x2)
        x3 = self.bn3(x3)
        x3 = torch.relu(x3)
        x3 = self.dropout3(x3)

        # Another residual connection
        x3 = x3 + self.residual_fc3(x2)  # Project x2 if needed

        # Output layer with inflation factor scaling
        return self.fc_out(x3) * inflation_factor

def split_data(X_train_tensor, y_train_tensor, validation_split=0.2, seed=42):
    """
    Splits the dataset into training and validation sets using PyTorch's random_split.
    """
    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)  # Combine X and y
    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size

    torch.manual_seed(seed)  # Ensure reproducibility
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset

def train_model(model, train_loader, val_loader, epochs=2000, batch_size=256, patience=10):
    """
    Train the model with early stopping and learning rate scheduling.
    """
    if (len(train_loader) * len(val_loader)) == 0:
        raise ValueError("training or/and validatation sets are empty")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Track losses for each epoch
    train_losses = []
    val_losses = []

    prev_val_loss = torch.tensor(float('inf'))

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                val_outputs = model(batch_x)
                val_loss = criterion(val_outputs, batch_y)
                total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)

        # Check for overfitting
        if avg_val_loss > prev_val_loss:
            print(f"Validation loss {avg_val_loss:.6f} increased at epoch {epoch}, consider stopping.")
        prev_val_loss = avg_val_loss

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save losses for each epoch
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Check early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 50 == 0:
            print(f"Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


def save_model(model, dummy_input, model_path_onnx, model_path_pth):
    """
    Save the trained model to ONNX format.
    """
    torch.onnx.export(model,
                      dummy_input,
                      model_path_onnx,
                      input_names=["input_name"],
                      output_names=["output_name"],
                      opset_version=12,
                      dynamic_axes={'input_name': {0: 'batch_size'},
                                    'output_name': {0: 'batch_size'}})
    print(f"Model saved to {model_path_onnx}")

    # Saving the model as a PyTorch model (.pth)
    torch.save(model.state_dict(), model_path_pth)
    print(f"PyTorch Model saved to {model_path_pth}")

def main():
    parser = argparse.ArgumentParser(description="Train and save a model for mesh deformation prediction.")
    parser.add_argument("input_file", type=str, help="Path to the input data file. (.pt)")
    parser.add_argument("output_file", type=str, help="Path to save the trained model (.onnx)")
    args = parser.parse_args()

    training_file = args.input_file.replace('\\', '/')
    save_file = args.output_file.replace('\\', '/')
    X_train_tensor, y_train_tensor = load_data(training_file)

    # Initialize model
    input_size = X_train_tensor.shape[1]  # Assuming the input tensor has shape (batch_size, input_size)
    model = DeltaPredictionModel(input_size)

    # Split data into training and validation sets

    # make sure we get the features that we need to train on
    # collection inside torus out, up, down torus

    train_dataset, val_dataset  = split_data(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Train the model
    model, _, _ = train_model(model, train_loader, val_loader)

    # Save the trained model to ONNX
    dummy_input = X_train_tensor[0].unsqueeze(0)  # Create a dummy input for the ONNX export
    dir_name, file_name = os.path.split(save_file)
    file_name, _ = file_name.split('.')
    pth_file_path = f"{dir_name}/{file_name}.pth"
    save_model(model, dummy_input, save_file, pth_file_path)

if __name__ == "__main__":
    main()
