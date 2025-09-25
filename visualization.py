def visualize_loss_landscape_3d(model, criterion, loader, device, p1=0.1, p2=0.1):
    import torch
    import matplotlib.pyplot as plt
    from torch.nn.utils import parameters_to_vector, vector_to_parameters
    from mpl_toolkits.mplot3d import Axes3D
    """Visualize the loss landscape in 3D by perturbing model parameters.

    Args:
        model: The neural network model.
        criterion: Loss function (e.g., nn.CrossEntropyLoss()).
        loader: DataLoader providing (X, y) mini-batches.
        device: torch.device to run the computation on.
        p1: Magnitude of perturbation in direction 1.
        p2: Magnitude of perturbation in direction 2.

    Returns:
        None. Displays a 3D surface plot of the loss landscape.
    """
    model.eval()

    # Flatten model parameters
    original_params = parameters_to_vector(model.parameters()).detach()
    direction1 = torch.randn_like(original_params)
    direction2 = torch.randn_like(original_params)

    # Normalize directions
    direction1 /= direction1.norm()
    direction2 /= direction2.norm()

    # Create a grid of perturbations
    grid_size = 21
    losses = torch.zeros((grid_size, grid_size))
    perturb_range = torch.linspace(-p1, p1, grid_size)

    for i, alpha in enumerate(perturb_range):
        for j, beta in enumerate(perturb_range):
            perturbed_params = original_params + alpha * direction1 + beta * direction2
            vector_to_parameters(perturbed_params, model.parameters())

            # Compute loss
            total_loss = 0.0
            total_samples = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                total_loss += loss.item() * xb.size(0)
                total_samples += xb.size(0)

            losses[i, j] = total_loss / total_samples

    # Restore original parameters
    vector_to_parameters(original_params, model.parameters())

    # Plot the loss landscape in 3D
    X, Y = torch.meshgrid(perturb_range, perturb_range, indexing="ij")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X.numpy(), Y.numpy(), losses.numpy(), cmap="viridis", edgecolor='none')
    ax.set_xlabel("Direction 1")
    ax.set_ylabel("Direction 2")
    ax.set_zlabel("Loss")
    ax.set_title("3D Loss Landscape")
    plt.show()


def visualize_loss_landscape_2d(model, criterion, loader, device, p1=0.1, p2=0.1):
    import matplotlib.pyplot as plt
    import torch
    from torch.nn.utils import parameters_to_vector, vector_to_parameters
    """Visualize the loss landscape by perturbing model parameters.

    Args:
        model: The neural network model.
        criterion: Loss function (e.g., nn.CrossEntropyLoss()).
        loader: DataLoader providing (X, y) mini-batches.
        device: torch.device to run the computation on.
        p1: Magnitude of perturbation in direction 1.
        p2: Magnitude of perturbation in direction 2.

    Returns:
        None. Displays a 2D surface plot of the loss landscape.
    """
    model.eval()

    # Flatten model parameters
    original_params = parameters_to_vector(model.parameters()).detach()
    direction1 = torch.randn_like(original_params)
    direction2 = torch.randn_like(original_params)

    # Normalize directions
    direction1 /= direction1.norm()
    direction2 /= direction2.norm()

    # Create a grid of perturbations
    grid_size = 21
    losses = torch.zeros((grid_size, grid_size))
    perturb_range = torch.linspace(-p1, p1, grid_size)

    for i, alpha in enumerate(perturb_range):
        for j, beta in enumerate(perturb_range):
            perturbed_params = original_params + alpha * direction1 + beta * direction2
            vector_to_parameters(perturbed_params, model.parameters())

            # Compute loss
            total_loss = 0.0
            total_samples = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                total_loss += loss.item() * xb.size(0)
                total_samples += xb.size(0)

            losses[i, j] = total_loss / total_samples

    # Restore original parameters
    vector_to_parameters(original_params, model.parameters())

    # Plot the loss landscape
    X, Y = torch.meshgrid(perturb_range, perturb_range, indexing="ij")
    plt.figure(figsize=(10, 8))
    plt.contourf(X.numpy(), Y.numpy(), losses.numpy(), levels=50, cmap="viridis")
    plt.colorbar(label="Loss")
    plt.xlabel("Direction 1")
    plt.ylabel("Direction 2")
    plt.title("Loss Landscape")
    plt.show()

def random_loss(model, criterion, loader, device, num_trials=100):
    """
    Randomize the weights of the model, calculate the loss and accuracy, and restore the original weights.

    Args:
        model: PyTorch model.
        criterion: Loss function.
        loader: DataLoader for evaluation.
        device: torch.device.
        num_trials: Number of random weight initializations to test.

    Returns:
        List of dictionaries with keys 'loss' and 'accuracy' for each random weight initialization.
    """
    import copy
    import torch
    # Save the original state of the model
    original_state = copy.deepcopy(model.state_dict())

    results = {
        "loss": [],
        "accuracy": []
    }

    for _ in range(num_trials):
        # Randomize the weights
        for param in model.parameters():
            if param.requires_grad:
                param.data.uniform_(-1.0, 1.0)  # Random values in range [-1, 1]

        # Calculate the loss and accuracy
        total_loss = 0.0
        total_samples = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                total_loss += loss.item() * xb.size(0)
                total_samples += xb.size(0)

                # Calculate accuracy
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()

        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        results["loss"].append(avg_loss)
        results["accuracy"].append(accuracy)

    # Restore the original weights
    model.load_state_dict(original_state)

    return results




