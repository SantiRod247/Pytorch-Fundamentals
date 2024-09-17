import torch


# Create a random tensor with shape (7, 7)
tensor = torch.rand(7, 7)
print(tensor)
print(tensor.shape)
print("\n")


# Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7) (hint: you may have to transpose the second tensor).
tensor2 = torch.rand(1, 7)
print(tensor2)
print(torch.matmul(tensor, tensor2.T))
print("\n")


# Set the random seed to 0 and do exercises 2 & 3 over again.
torch.manual_seed(0)

tensorr = torch.rand(7, 7)
print(tensorr)
print(tensorr.shape)

tensor2r = torch.rand(1, 7)
print(tensor2r)
print(torch.matmul(tensor, tensor2r.T))
print("\n")


# Speaking of random seeds, we saw how to set it with torch.manual_seed() but is there a GPU equivalent? 
# (hint: you'll need to look into the documentation for torch.cuda for this one). 
# If there is, set the GPU random seed to 1234.

torch.cuda.manual_seed(1234)
random_tensor_gpu = torch.randn(3, device='cuda')
print("numbers for GPU is -1.6165,  0.5685, -0.5102: ", random_tensor_gpu)
# there is a GPU equivalent for setting the random seed in PyTorch using torch.cuda. 
# You can set the random seed for CUDA (GPU) operations using torch.cuda.manual_seed()
print("\n")


# Create two random tensors of shape (2, 3) and send them both to the GPU (you'll need access to a GPU for this). 
# Set torch.manual_seed(1234) when creating the tensors (this doesn't have to be the GPU random seed).
torch.manual_seed(1234)
tensor_1 = torch.rand(2, 3)
tensor_2 = torch.rand(2, 3)
print(tensor_1)
print(tensor_2)

# Check if GPU is available and send tensors to the GPU
if torch.cuda.is_available():
    tensor_1 = tensor_1.to('cuda')
    tensor_2 = tensor_2.to('cuda')
    print("Tensor 1 on GPU:", tensor_1)
    print("Tensor 2 on GPU:", tensor_2)
else:
    print("GPU not available. Tensors remain on CPU.")
    print("Tensor 1:", tensor_1)
    print("Tensor 2:", tensor_2)
print("\n")


# Perform a matrix multiplication on the tensors you created in 6 
# (again, you may have to adjust the shapes of one of the tensors).
output_7 = torch.matmul(tensor_1, tensor_2.T)
print(output_7)
print("\n")


# Find the maximum and minimum values of the output of 7
print(f"Minimum: {output_7.min()}")
print(f"Maximum: {output_7.max()}")
print("\n")

# Find the maximum and minimum index values of the output of 7.
print(f"Index where max value occurs: {output_7.argmax()}")
print(f"Index where min value occurs: {output_7.argmin()}")
print("\n")

# Make a random tensor with shape (1, 1, 1, 10) and then create a new 
# tensor with all the 1 dimensions removed to be left with a tensor of shape (10). 
# Set the seed to 7 when you create it and print out the first tensor and it's shape 
# as well as the second tensor and it's shape.
torch.manual_seed(7)
last_tensor = torch.rand(1, 1, 1, 10)
last_tensor_squeeze = torch.squeeze(last_tensor)
print(last_tensor)
print(last_tensor_squeeze)
print("\n")
