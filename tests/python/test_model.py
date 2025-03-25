import os
import torch
import pytest
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the 'python' folder to the sys.path for module resolution
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, f"{root_path}/python")

import onnxDeformer.train_model as tm


# Test for loading data
@pytest.mark.parametrize("data_path", [f"{root_path}/data/training_data.pt"])
def test_load_data(data_path):
    X_train, y_train = tm.load_data(data_path)
    assert (
        X_train.shape[0] == y_train.shape[0]
    ), "Number of samples in X and y do not match"


def test_data_split():
    X_train_tensor = torch.randn(1000, 4)  # Dummy data for testing
    y_train_tensor = torch.randn(1000, 4)  # Dummy data for testing
    train_dataset, val_dataset = tm.split_data(X_train_tensor, y_train_tensor)
    assert len(train_dataset) + len(val_dataset) == 1000, "Incorrect split size"


@pytest.fixture
def model_and_data():
    """
    Fixture to create a model and dummy data for tests.
    """
    # Dummy data for testing
    X_train_tensor = torch.randn(1000, 7)
    y_train_tensor = torch.randn(1000, 3)

    # Create model
    input_size = X_train_tensor.shape[1]
    model = tm.DeltaPredictionModel(input_size)

    return model, X_train_tensor, y_train_tensor


def test_training_with_fixture(model_and_data):
    model, X_train_tensor, y_train_tensor = model_and_data
    train_dataset, val_dataset = tm.split_data(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)

    model, _, _ = tm.train_model(model, train_loader, val_loader, epochs=2)
    # After 2 epochs, the model should have been trained without errors
    assert isinstance(model, tm.DeltaPredictionModel), "Model is not the correct type"


def test_save_model_with_fixture(model_and_data):
    model, _, _ = model_and_data
    dummy_input = torch.randn(1, 7)  # Dummy input
    model_path = "model.onnx"
    tm.save_model(model, dummy_input, model_path, "model.pth")

    # Check if the file was saved
    assert os.path.exists(model_path), "Model file was not saved"

    # Clean up after test
    if os.path.exists(model_path):
        os.remove(model_path)


# --------------------
#  Test Edge Cases
# --------------------


def test_invalid_data_path():
    invalid_path = "invalid_path.pt"
    with pytest.raises(FileNotFoundError):
        tm.load_data(invalid_path)


def test_model_training_no_data():
    model = tm.DeltaPredictionModel(0)
    empty_train = torch.randn(0, 4)
    empty_val = torch.randn(0, 3)

    with pytest.raises(ValueError):
        tm.train_model(model, empty_train, empty_val)
