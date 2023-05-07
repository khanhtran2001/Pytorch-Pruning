from tqdm.auto import tqdm, trange
from tqdm import tqdm
import torch
from data import trainset,valset,testset
from models import MobileNetV3
from utils import smoothed_l0_regularize


# Load data
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)
val_loader = torch.utils.data.DataLoader(valset, batch_size=32,
                                         shuffle=False, num_workers=2)


# Define model
model = MobileNetV3(mode='small', classes_num=10, input_size=32, width_multiplier=1)

# Define the loss function as cross entropy loss
criterion = torch.nn.CrossEntropyLoss()

# Check if the device supports GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model.to(device)

# Get the list of layers to prune
layers_to_prune = [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]
# Define the hyperparameters for retraining
num_epochs = 100
learning_rate = 0.001
lamb = 1e-7
c = 10

# Define the optimizer and the learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Set the model to training mode
model.train()

# Loop over the epochs
progress_bar = tqdm(range(num_epochs))

def train_pruning():
    loss_l0 = 0
    for epoch in progress_bar:
        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_acc = 0.0
        # Loop over the batches in the training data
        for inputs, labels in tqdm(train_loader):
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Clear the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            # Get weights according to layers
            for layer in layers_to_prune:
                W = layer.weight.data.to(device)  # Move the weight matrix to the device
                loss_l0 += smoothed_l0_regularize(W, c, lamb)
            # Compute the loss
            loss = criterion(outputs, labels) + loss_l0
            # Backward pass
            loss.backward()
            # Update the weights
            optimizer.step()
            # Compute the accuracy and accumulate it
            _, preds = torch.max(outputs, 1)
            acc = torch.mean((preds == labels).float())
            running_acc += acc.item()
            # Accumulate the loss
            running_loss += loss.item()
        # Update the learning rate
        scheduler.step()
        # Compute the average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)
        # Print the loss and accuracy for the epoch
        progress_bar.set_description(
            f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}"
        )


if __name__ == "__main__":
    train_pruning()
