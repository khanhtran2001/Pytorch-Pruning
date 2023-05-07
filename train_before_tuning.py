import torch
from data import trainset,testset,valset
from models import MobileNetV3
from utils import get_model_parameters,_ensure_divisible


# Load data
# Create data loaders for train, test and validation sets
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

# Cell 3: Check and use GPU
# Check if the device supports GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model.to(device)

# Cell 4: Train the model with CIFAR-10 before pruning
# Define the number of epochs and the learning rate
num_epochs = 50
lr = 0.001

# Define the optimizer as SGD
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

def train():
    print(f"Using device: {device}")
    # Train the model for num_epochs
    for epoch in range(num_epochs):
        # Initialize the variables to store the train loss and accuracy
        train_loss = 0.0
        train_acc = 0.0
       
        # Loop over the training data
        for inputs, labels in train_loader:
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            # Compute the loss
            loss = criterion(outputs, labels)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            # Accumulate the loss and accuracy
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            acc = torch.mean((preds == labels).float())
            train_acc += acc.item()
        # Compute the average train loss and accuracy
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        # Print the epoch, loss and accuracy
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")


if __name__ == "__main__":
    train()
