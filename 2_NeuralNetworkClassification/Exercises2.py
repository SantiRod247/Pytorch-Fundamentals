import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import plot_predictions, plot_decision_boundary
from sklearn.datasets import make_moons

# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print("\n")

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

# Make a binary classification dataset with Scikit-Learn's make_moons() function.
# For consistency, the dataset should have 1000 samples and a random_state=42.
# Turn the data into PyTorch tensors. Split the data into training and test sets 
# using train_test_split with 80% training and 20% testing.

# Make 1000 samples 
n_samples = 1000
# Create circles
X, y = make_moons(n_samples,
                    noise=0.03,
                    random_state=42)

# Visualize with a plot
plt.figure(figsize=(10, 7))
plt.scatter(x=X[:, 0], 
            y=X[:, 1], 
            c=y, 
            cmap=plt.cm.RdYlBu)
# Save the plot as an image 
plt.savefig('2_NeuralNetworkClassification/Ex_moons_dataset.png')
print("Plot saved as 'Ex_moons_dataset.png'")
plt.clf()

# Turn data into tensors
# Otherwise this causes issues with computations later on
import torch
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# View the first five samples
print(X[:5], y[:5], "\n")

# Split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible

print(len(X_train), len(X_test), len(y_train), len(y_test))
print("\n")


# Build a model by subclassing nn.Module that incorporates non-linear activation 
# functions and is capable of fitting the data you created in 1.

# Build model with non-linear activation function
from torch import nn
class MoonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
      # Intersperse the ReLU activation function between layers
       return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_moon = MoonModel().to(device)
print(model_moon)
print("\n")


# Setup a binary classification compatible loss function and optimizer to use when 
# training the model.

# loss_fn = nn.BCELoss() # Requires sigmoid on input
loss_fn = nn.BCEWithLogitsLoss() # Does not require sigmoid on input
optimizer = torch.optim.SGD(model_moon.parameters(), lr=0.1)

# Create a training and testing loop to fit the model you created in 2 to the data you created in 1.
# Train the model for long enough for it to reach over 96% accuracy.
# The training loop should output progress every 10 epochs of the model's training and test set loss and accuracy.

# Fit the model
torch.manual_seed(42)
epochs = 600

# Put all data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    # 1. Forward pass
    y_logits = model_moon(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels
    
    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss calculates loss using logits
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred)
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_moon.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_logits = model_moon(X_test).squeeze()
      test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
      # 2. Calculate loss and accuracy
      test_loss = loss_fn(test_logits, y_test)
      test_acc = accuracy_fn(y_true=y_test,
                             y_pred=test_pred)

    # Print out what's happening
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")
print("\n")      


# Make predictions with your trained model and plot them using the 
# plot_decision_boundary() function created in this notebook.

import requests
from pathlib import Path 
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_moon, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_moon, X_test, y_test)
plt.savefig('2_NeuralNetworkClassification/Ex_trainandtest.png')
print("Plot saved as 'Ex_trainandtest.png'")
print("\n")      


# Replicate the Tanh (hyperbolic tangent) activation function in pure PyTorch.

class MoonModelTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.tanh = nn.Tanh() 

    def forward(self, x):
      # Intersperse the ReLU activation function between layers
       return self.layer_3(self.tanh(self.layer_2(self.tanh(self.layer_1(x)))))

model_moon_tanh = MoonModel().to(device)
print(model_moon_tanh)
print("\n")

optimizer_tanh = torch.optim.SGD(model_moon_tanh.parameters(), lr=0.1)

for epoch in range(epochs):
    # 1. Forward pass
    y_logits = model_moon_tanh(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels
    
    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss calculates loss using logits
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred)
    
    # 3. Optimizer zero grad
    optimizer_tanh.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer_tanh.step()

    ### Testing
    model_moon_tanh.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_logits = model_moon_tanh(X_test).squeeze()
      test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
      # 2. Calculate loss and accuracy
      test_loss = loss_fn(test_logits, y_test)
      test_acc = accuracy_fn(y_true=y_test,
                             y_pred=test_pred)

    # Print out what's happening
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_moon_tanh, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_moon_tanh, X_test, y_test)
plt.savefig('2_NeuralNetworkClassification/Ex_trainandtestTanh.png')
print("Plot saved as 'Ex_trainandtestTanh.png'")
plt.clf()  # Clear the current plot
print("\n")  