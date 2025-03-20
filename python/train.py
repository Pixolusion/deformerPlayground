from maya.api import OpenMaya as om2
from maya import cmds
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Open the Maya file
cmds.file("P:/Code/github/deformerCompare/data/torusTest.mb", open=True, force=True)

def get_point_positions_and_deltas(mesh_name, deformer_name, base_mesh_name=None):
    """
    Get the current mesh positions and the deformation deltas.
    If base_mesh_name is provided, calculate deltas from that mesh.
    Otherwise, use the original positions from the mesh's history.
    """
    # Get the deformed mesh
    selection_list = om2.MSelectionList()
    selection_list.add(mesh_name)
    mesh_obj = selection_list.getDagPath(0)

    # Get MFnMesh object for the deformed mesh
    mesh_fn = om2.MFnMesh(mesh_obj)

    # Get the deformed positions
    deformed_points = mesh_fn.getPoints(om2.MSpace.kWorld)

    # Get the original positions (either from base mesh or history)
    if base_mesh_name:
        # Use provided base mesh
        base_selection = om2.MSelectionList()
        base_selection.add(base_mesh_name)
        base_mesh_obj = base_selection.getDagPath(0)
        base_mesh_fn = om2.MFnMesh(base_mesh_obj)
        original_points = base_mesh_fn.getPoints(om2.MSpace.kWorld)
    else:
        # Try to get original points from history
        # This is a simplified approach - in production, you'd need more robust history access
        original_points = mesh_fn.getPoints(om2.MSpace.kObject)

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
base_mesh_name = "pTorusShape1Orig"
deformer_name = "inflateDeformer1"
start_frame = 1
end_frame = 100

# Collect data from multiple frames
X_train_list = []
y_train_list = []

for frame in range(start_frame, end_frame + 1):
    cmds.currentTime(frame)  # Set current frame in Maya
    points, deltas, inflation_factor = get_point_positions_and_deltas(mesh_name, deformer_name, base_mesh_name)

    # Store each point's position, inflation factor, and corresponding delta
    for i, point in enumerate(points):
        # Input features: x, y, z, inflation_factor
        X_train_list.append([point.x, point.y, point.z, inflation_factor])

        # Output: delta x, delta y, delta z
        y_train_list.append([deltas[i][0], deltas[i][1], deltas[i][2]])

# Convert collected data to NumPy arrays
X_train = np.array(X_train_list, dtype=np.float32)
y_train = np.array(y_train_list, dtype=np.float32)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)

# Define the model architecture
class DeltaPredictionModel(nn.Module):
    def __init__(self):
        super(DeltaPredictionModel, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # Input: 4 features (x, y, z, inflation_factor)
        self.fc2 = nn.Linear(128, 64)  # Hidden layer
        self.fc3 = nn.Linear(64, 3)    # Output: 3 values for delta_x, delta_y, delta_z

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = DeltaPredictionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
batch_size = 256
epochs = 1000

for epoch in range(epochs):
    model.train()

    # Mini-batch training
    permutation = torch.randperm(X_train_tensor.size()[0])
    total_loss = 0

    for i in range(0, X_train_tensor.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {total_loss / (X_train_tensor.size()[0] / batch_size)}")

# Save the trained model to ONNX format
# Create a dummy input with the correct shape
dummy_input = torch.randn(1, 4)  # Batch size of 1, 4 features per point

torch.onnx.export(model,
                  dummy_input,
                  "P:/Code/github/deformerCompare/data/delta_prediction_model.onnx",
                  input_names=["input_name"],
                  output_names=["output_name"],
                  opset_version=12,
                  dynamic_axes={'input_name': {0: 'batch_size'},
                                'output_name': {0: 'batch_size'}})

print("Model training complete and saved as delta_prediction_model.onnx")
