# Import PyTorch
import torch
# Import matplotlib for visualization
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import accuracy metric
from helper_functions import accuracy_fn # Note: could also use torchmetrics.Accuracy(task = 'multiclass', num_classes=len(class_names)).to(device)

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
torch.manual_seed(24)

## necessary imports


# What are 3 areas in industry where computer vision is currently being used?

# 1. Autonomous Vehicles
# Autonomous vehicles use computer vision to interpret the environment, 
# identify traffic signs, detect pedestrians, other vehicles and obstacles, 
# and make real-time decisions for safe driving.

# 2. Security
# Computer vision is used in surveillance systems to recognize faces, 
# detect unusual or dangerous behavior, and track objects or people in real time.

# 3. Health (Medical Imaging)
# In medicine, computer vision is used to analyze medical images, 
# such as X-rays, CT scans or MRI scans, helping doctors to detect diseases, 
# such as cancer or abnormalities.


# Search "what is overfitting in machine learning" and write down a sentence about what you find.

# Overfitting in machine learning occurs when a model fits the training data too well, 
# but does not generalize well to new data or to a test set. That is, 
# the model learns specific details and noise from the training set that are not relevant or representative of new data. 
# This leads to poor performance in predicting new examples. 


# Search "ways to prevent overfitting in machine learning", write down 3 of the things you find and a sentence about each.

# 1. Increase the size of the data set: 
# Collect more real data or use data enhancement techniques 
# (such as rotating, flipping or adjusting the brightness of images) to generate more training examples

# 2. Regularization (L1/L2):
# Adds a penalty to the loss function based on the values of the model weights, helping to prevent the weights from growing too large. 
# L2 (Ridge Regularization): Penalizes the square of large weights, helping to distribute the parameter values more evenly. 
# L1 (Lasso Regularization): Penalizes the absolute values of the weights, which can force some parameters to be exactly 0, generating simpler models.

# 3. Dropout:
# Randomly drops out (sets to zero) some of the neurons in the network during training. 
# This forces the network to learn more robust features and reduces the risk of overfitting.


# Load the torchvision.datasets.MNIST() train and test datasets.
# Create a dataloader for the training and testing datasets

# Setup training data
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False, # get test data
    download=True,
    transform=ToTensor()
)


# Visualize at least 5 different samples of the MNIST training dataset.

fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(label)
    plt.axis(False)
plt.savefig('3_ComputerVision/Exercises3_Visualizingdata(1).png')
print("Plot saved as 'Exercises3_Visualizingdata(1).png'")
plt.clf()  # Clear the current plot


# Turn the MNIST train and test datasets into dataloaders using 
# torch.utils.data.DataLoader, set the batch_size=32.

# Setup the batch size hyperparameter
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch? 
    shuffle=True # shuffle data every epoch?
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False # don't necessarily have to shuffle the testing data
)
