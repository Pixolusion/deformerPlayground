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

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Load the collected data
X_train, y_train = torch.load("P:/Code/github/pixolusion/deformerPlayground/data/training_data.pt")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)

class DeltaPredictionModel(nn.Module):
    """
    A neural network model for predicting the deltas of mesh points based on their positions
    and the inflation factor.

    Args:
        dropout_rate (float): The dropout rate to apply to each dropout layer (default is 0.2).
    """
    def __init__(self, dropout_rate=0.2):
        super(DeltaPredictionModel, self).__init__()
        # Input layer
        self.fc1 = nn.Linear(4, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Hidden layers with residual connections
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Output layer
        self.fc_out = nn.Linear(128, 3)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (Tensor): Input tensor containing the point positions and inflation factor (batch_size, 4).

        Returns:
            Tensor: The predicted deltas (delta_x, delta_y, delta_z) for each point in the batch.
        """
        # First block
        inflation_factor = x[:, 3].unsqueeze(1)
        x1 = torch.relu(self.bn1(self.fc1(x)))
        x1 = self.dropout1(x1)

        # Residual block 1
        x2 = torch.relu(self.bn2(self.fc2(x1)))
        x2 = self.dropout2(x2) + x1

        # Residual block 2
        x3 = torch.relu(self.bn3(self.fc3(x2)))
        x3 = self.dropout3(x3) + x2

        # Predict deltas and scale by inflation
        return self.fc_out(x3) * inflation_factor

# Initialize model
model = DeltaPredictionModel()
criterion = nn.L1Loss()
# Added weight decay to prevent overfitting
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Training loop with early stopping
batch_size = 256
epochs = 2000
best_loss = float('inf')
patience = 20
patience_counter = 0
best_model_state = None

# Split data into training and validation sets
validation_split = 0.2
indices = list(range(len(X_train_tensor)))
random.shuffle(indices)
split = int(np.floor(validation_split * len(X_train_tensor)))
train_indices, valid_indices = indices[split:], indices[:split]

for epoch in range(epochs):
    """
    Training loop for the neural network model.
    """
    model.train()
    permutation = torch.randperm(len(train_indices))
    total_train_loss = 0

    for i in range(0, len(train_indices), batch_size):
        batch_indices = [train_indices[idx] for idx in permutation[i:i+batch_size]]
        batch_x, batch_y = X_train_tensor[batch_indices], y_train_tensor[batch_indices]
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / (len(train_indices) / batch_size)

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_train_tensor[valid_indices])
        val_loss = criterion(val_outputs, y_train_tensor[valid_indices])

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Print progress
    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

    if val_loss < best_loss:
        best_loss = val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Save the trained model to ONNX format
# Create a dummy input with the correct shape
dummy_input = torch.randn(1, 4)  # Batch size of 1, 4 features per point
model_path = "P:/Code/github/pixolusion/deformerPlayground/data/delta_model_L1Loss.onnx"
torch.onnx.export(model,
                  dummy_input,
                  model_path,
                  input_names=["input_name"],
                  output_names=["output_name"],
                  opset_version=12,
                  dynamic_axes={'input_name': {0: 'batch_size'},
                                'output_name': {0: 'batch_size'}})
print("Model training complete and saved.")
