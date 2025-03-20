from maya.api import OpenMaya as om2
from maya import cmds
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Open the Maya file
cmds.file("P:/Code/github/deformerCompare/data/torusTest.mb", open=True, force=True)

def get_point_positions_and_inflation_factor(name, deformer_name):
    # Get the MObject for the mesh and the deformer
    selection_list = om2.MSelectionList()
    selection_list.add(name)
    mesh_obj = selection_list.getDagPath(0)

    # Get MFnMesh object to access mesh data
    mesh_fn = om2.MFnMesh(mesh_obj)

    # Get the positions of the mesh vertices (points)
    points = mesh_fn.getPoints(om2.MSpace.kWorld)  # Get world space coordinates

    # Extract the inflation amount from the deformer
    selection_list.add(deformer_name)
    deformer_obj = selection_list.getDependNode(1)
    deformer_fn = om2.MFnDependencyNode(deformer_obj)

    # Get the inflation attribute
    inflate_attr = deformer_fn.attribute("inflateAmount")
    inflate_amount_plug = om2.MPlug(deformer_obj, inflate_attr)
    inflate_amount = inflate_amount_plug.asFloat()

    return points, inflate_amount

# Configuration
mesh_name = "pTorusShape1"
deformer_name = "inflateDeformer1"
start_frame = 1
end_frame = 100

# Get initial point length for reference
points_tmp, _ = get_point_positions_and_inflation_factor(mesh_name, deformer_name)
point_len = len(points_tmp)

# Collect data from multiple frames
X_train_list = []
y_train_list = []

for frame in range(start_frame, end_frame + 1):
    cmds.currentTime(frame)  # Set current frame in Maya
    points, inflation_factor = get_point_positions_and_inflation_factor(mesh_name, deformer_name)

    # Store each point's position and the inflation factor
    for point in points:
        # Input features: x, y, z, inflation_factor
        X_train_list.append([point.x, point.y, point.z, inflation_factor])

        # Output: We want to predict the same inflation factor
        y_train_list.append([inflation_factor])

# Convert collected data to NumPy arrays
X_train = np.array(X_train_list, dtype=np.float32)
y_train = np.array(y_train_list, dtype=np.float32)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)

# Define the model architecture
class InflationPredictionModel(nn.Module):
    def __init__(self):
        super(InflationPredictionModel, self).__init__()
        self.fc1 = nn.Linear(4, 64)  # Input: 5 features (x, y, z, inflation_factor)
        self.fc2 = nn.Linear(64, 32)  # Hidden layer
        self.fc3 = nn.Linear(32, 1)   # Output: 1 predicted inflation factor

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = InflationPredictionModel()
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
                  "P:/Code/github/deformerCompare/data/inflation_model.onnx",
                  input_names=["input_name"],
                  output_names=["output_name"],
                  opset_version=12,
                  dynamic_axes={'input_name': {0: 'batch_size'},
                                'output_name': {0: 'batch_size'}})

print("Model training complete and saved as inflation_model.onnx")