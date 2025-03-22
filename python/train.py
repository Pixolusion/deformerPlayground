from maya.api import OpenMaya as om2
from maya import cmds
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Open the Maya file
cmds.file("P:/Code/github/pixolusion/deformerCompare/data/torusTest.mb", open=True, force=True)

def get_point_positions_and_deltas(mesh_name, deformer_name, base_mesh_name):
    """
    Extracts the positions of mesh points, their deformation deltas, and the inflation factor from the Maya scene.

    Args:
        mesh_name (str): Name of the mesh in the Maya scene.
        deformer_name (str): Name of the deformer applied to the mesh.
        base_mesh_name (str): Name of the base mesh to use for comparison.

    Returns:
        tuple: A tuple containing:
            - original_points (list of MPoint): The original positions of the points in the mesh.
            - deltas (list of tuple): The deformation deltas for each point (delta_x, delta_y, delta_z).
            - inflate_amount (float): The inflation factor applied by the deformer.
    """
    # Get the deformed mesh
    selection_list = om2.MSelectionList()
    selection_list.add(mesh_name)
    mesh_obj = selection_list.getDagPath(0)

    # Get MFnMesh object for the deformed mesh
    mesh_fn = om2.MFnMesh(mesh_obj)

    # Get the deformed positions
    deformed_points = mesh_fn.getPoints(om2.MSpace.kWorld)

    # Get the original positions
    base_selection = om2.MSelectionList()
    base_selection.add(base_mesh_name)
    base_mesh_obj = base_selection.getDagPath(0)
    base_mesh_fn = om2.MFnMesh(base_mesh_obj)
    original_points = base_mesh_fn.getPoints(om2.MSpace.kWorld)

    # Extract the inflation amount from the deformer
    selection_list.add(deformer_name)
    deformer_obj = selection_list.getDependNode(1)
    deformer_fn = om2.MFnDependencyNode(deformer_obj)

    # Get the inflation attribute
    inflate_attr = deformer_fn.attribute("inflateAmount")
    inflate_amount_plug = om2.MPlug(deformer_obj, inflate_attr)
    inflate_amount = inflate_amount_plug.asFloat()

    # Calculate deltas
    deltas = [(deformed_points[i].x - original_points[i].x,
               deformed_points[i].y - original_points[i].y,
               deformed_points[i].z - original_points[i].z)
              for i in range(len(deformed_points))]

    return original_points, deltas, inflate_amount

# Configuration
mesh_name = "pTorusShape1"
deformer_name = "inflateDeformer1"
base_mesh_name = "pTorusShape1Orig"
start_frame = 1
end_frame = 120

# Collect data from multiple frames
X_train_list = []
y_train_list = []
inflation_values = []
points, deltas, inflation_factor = get_point_positions_and_deltas(mesh_name, deformer_name, base_mesh_name)

# Add explicit zero-inflation data
for i, point in enumerate(points):
    X_train_list.append([point.x, point.y, point.z, 0.0])  # Inflation factor 0
    y_train_list.append([0.0, 0.0, 0.0])  # No deformation

for frame in range(start_frame, end_frame + 1):
    cmds.currentTime(frame)  # Set current frame in Maya
    points, deltas, inflation_factor = get_point_positions_and_deltas(mesh_name, deformer_name, base_mesh_name)
    inflation_values.append(inflation_factor)

    # Store each point's position, inflation factor, and corresponding delta
    for i, point in enumerate(points):
        # Input features: normalized x, y, z, normalized inflation_factor
        X_train_list.append([
            point.x,
            point.y,
            point.z,
            inflation_factor
        ])

        # Output: delta x, delta y, delta z
        y_train_list.append([deltas[i][0], deltas[i][1], deltas[i][2]])

# to help overfitting we augment some data to introduce diversity
# Slightly move points for generalization
for i, point in enumerate(points):
    perturb = np.random.uniform(-0.01, 0.01, 3)  # Small noise
    X_train_list.append([point.x + perturb[0], point.y + perturb[1], point.z + perturb[2], inflation_factor])
    y_train_list.append([deltas[i][0], deltas[i][1], deltas[i][2]])

# Get a subset of representative points for synthetic data generation
points_subset = random.sample(list(points), min(500, len(points)))

# Convert collected data to NumPy arrays
X_train = np.array(X_train_list, dtype=np.float32)
y_train = np.array(y_train_list, dtype=np.float32)

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
        self.fc_out = nn.Linear(128, 3)  # Output: 3 values for delta_x, delta_y, delta_z

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (Tensor): Input tensor containing the point positions and inflation factor (batch_size, 4).

        Returns:
            Tensor: The predicted deltas (delta_x, delta_y, delta_z) for each point in the batch.
        """
        # First block
        inflation_factor = x[:, 3].unsqueeze(1)  # Extract inflation as a separate tensor
        x1 = torch.relu(self.bn1(self.fc1(x)))
        x1 = self.dropout1(x1)

        # Residual block 1
        x2 = torch.relu(self.bn2(self.fc2(x1)))
        x2 = self.dropout2(x2)
        x2 = x2 + x1  # Residual connection

        # Residual block 2
        x3 = torch.relu(self.bn3(self.fc3(x2)))
        x3 = self.dropout3(x3)
        x3 = x3 + x2  # Residual connection

        # Predict deltas and scale by inflation
        out = self.fc_out(x3) * inflation_factor  # Explicit inflation scaling

        return out

# Initialize the model, loss function, and optimizer
model = DeltaPredictionModel()

criterion = nn.L1Loss()
# Added weight decay to prevent overfitting
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Learning rate scheduler
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

    # Mini-batch training
    permutation = torch.randperm(len(train_indices))
    total_train_loss = 0

    for i in range(0, len(train_indices), batch_size):
        batch_indices = [train_indices[idx] for idx in permutation[i:i+batch_size]]
        batch_x, batch_y = X_train_tensor[batch_indices], y_train_tensor[batch_indices]

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimize
        loss.backward()

        # Gradient clipping to prevent exploding gradients
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

    # Early stopping check
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Load the best model state
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Save the trained model to ONNX format
# Create a dummy input with the correct shape
dummy_input = torch.randn(1, 4)  # Batch size of 1, 4 features per point
model_path = "P:/Code/github/pixolusion/deformerCompare/data/delta_model_L1Loss.onnx"
torch.onnx.export(model,
                  dummy_input,
                  model_path,
                  input_names=["input_name"],
                  output_names=["output_name"],
                  opset_version=12,
                  dynamic_axes={'input_name': {0: 'batch_size'},
                                'output_name': {0: 'batch_size'}})

print(f"Model training complete and saved as {model_path}")
