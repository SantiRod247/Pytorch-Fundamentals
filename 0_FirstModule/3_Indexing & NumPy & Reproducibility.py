import torch
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x, x.shape)

print("Let's index bracket by bracket")
print(f"First square bracket:\n{x[0]}") 
print(f"Second square bracket: {x[0][0]}") 
print(f"Third square bracket: {x[0][0][0]}")

print("Get all values of 0th dimension and the 0 index of 1st dimension")
print(x[:, 0])

print("Get all values of 0th & 1st dimensions but only index 1 of 2nd dimension")
print(x[:, :, 1])

print("Get all values of the 0 dimension but only the 1 index value of the 1st and 2nd dimension")
print(x[:, 1, 1])

print("Get index 0 of 0th and 1st dimension and all values of 2nd dimension")
print(x[0, 0, :]) # same as x[0][0]

print("NumPy to tensor")
import numpy as np
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(array, tensor)

print("Change the array, keep the tensor")
array = array + 1
print(array, tensor)

print("Tensor to NumPy array")
tensor = torch.ones(7) # create a tensor of ones with dtype=float32
numpy_tensor = tensor.numpy() # will be dtype=float32 unless changed
print(tensor, numpy_tensor)

print("Change the tensor, keep the array the same")
tensor = tensor + 1
print(tensor, numpy_tensor)

print("Create two random tensors")
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(f"Tensor A:\n{random_tensor_A}\n")
print(f"Tensor B:\n{random_tensor_B}\n")
print(f"Does Tensor A equal Tensor B? (anywhere)")
print(random_tensor_A == random_tensor_B)

import torch
import random

print("Set the random seed")
RANDOM_SEED=42 # try changing this to different values and see what happens to the numbers below
torch.manual_seed(seed=RANDOM_SEED) 
random_tensor_C = torch.rand(3, 4)

print("Have to reset the seed every time a new rand() is called")
print("Without this, tensor_D would be different to tensor_C")
torch.random.manual_seed(seed=RANDOM_SEED) # try commenting this line out and seeing what happens
random_tensor_D = torch.rand(3, 4)

print(f"Tensor C:\n{random_tensor_C}\n")
print(f"Tensor D:\n{random_tensor_D}\n")
print(f"Does Tensor C equal Tensor D? (anywhere)")
print(random_tensor_C == random_tensor_D)