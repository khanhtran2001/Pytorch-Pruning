# Cell 1: Import the modules and load the dataset
# Import the torch and torchvision modules
import torch
import torchvision

# Define the transform function to convert data to tensor
transform = torchvision.transforms.ToTensor()

# Download and load the CIFAR-10 dataset with transform
dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# Split the dataset into train, test and validation sets
train_size = 40000
test_size = 5000
val_size = 5000
trainset, testset, valset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

