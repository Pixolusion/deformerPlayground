import train_model as tm

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def plot_largest_error_predictions(model, X_test, y_test, error_threshold=0.2):
    """
    Plots the model's predictions for the points where the error is largest.
    """
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()

    y_test = y_test.cpu().numpy()
    # Calculate the error magnitude for each point
    errors = np.linalg.norm(y_pred - y_test, axis=1)

    # Select points with the largest errors
    large_error_indices = np.where(errors > error_threshold)[0]

    # If there are too many points with large error, reduce them
    if len(large_error_indices) > 100:
        large_error_indices = np.random.choice(
            large_error_indices, size=100, replace=False
        )

    y_test_subset = y_test[large_error_indices]
    y_pred_subset = y_pred[large_error_indices]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot true and predicted values for the subset with large errors
    ax.scatter(
        y_test_subset[:, 0],
        y_test_subset[:, 1],
        y_test_subset[:, 2],
        c="blue",
        label="True Values",
        alpha=0.5,
    )
    ax.scatter(
        y_pred_subset[:, 0],
        y_pred_subset[:, 1],
        y_pred_subset[:, 2],
        c="red",
        label="Predictions",
        alpha=0.5,
    )

    # Add lines between true and predicted values (errors)
    for i in range(len(y_test_subset)):
        ax.plot(
            [y_test_subset[i, 0], y_pred_subset[i, 0]],
            [y_test_subset[i, 1], y_pred_subset[i, 1]],
            [y_test_subset[i, 2], y_pred_subset[i, 2]],
            color="gray",
            alpha=0.5,
        )

    ax.set_xlabel("Delta X")
    ax.set_ylabel("Delta Y")
    ax.set_zlabel("Delta Z")
    ax.set_title("3D Model Predictions vs. Ground Truth (Large Errors)")
    ax.legend()
    plt.show()


def plot_loss(train_losses, val_losses):
    """
    Plots the training and validation loss curves.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss", color="blue", linestyle="-")
    plt.plot(val_losses, label="Validation Loss", color="red", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()


def show_largest_error_predictions():
    torch_train_data_path = f"{root_path}/data/training_data.pt"
    X_test, y_test = torch.load(torch_train_data_path)  # Load some test data
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)
    model_path = f"{root_path}/data/delta_model_L1Loss.pth"
    input_size = X_test.shape[
        1
    ]  # Assuming the input tensor has shape (batch_size, input_size)
    trained_model = tm.DeltaPredictionModel(input_size)
    trained_model.load_state_dict(torch.load(model_path))
    trained_model.eval()

    plot_largest_error_predictions(trained_model, X_test_tensor, y_test_tensor)


def show_loss():
    torch_train_data_path = f"{root_path}/data/training_data.pt"
    X_test_tensor, y_test_tensor = tm.load_data(torch_train_data_path)

    # Initialize model
    input_size = X_test_tensor.shape[
        1
    ]  # Assuming the input tensor has shape (batch_size, input_size)
    model = tm.DeltaPredictionModel(input_size)

    # Split data into training and validation sets
    train_dataset, val_dataset = tm.split_data(X_test_tensor, y_test_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Train the model
    model, train_losses, val_losses = tm.train_model(model, train_loader, val_loader)
    plot_loss(train_losses, val_losses)


if __name__ == "__main__":
    show_largest_error_predictions()
    # show_loss()
