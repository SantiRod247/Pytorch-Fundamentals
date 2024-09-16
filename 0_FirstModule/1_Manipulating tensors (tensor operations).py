import torch 

print("Scalar")
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim)
print(scalar.item())

print("Vector")
vector = torch.tensor([7, 7])
print(vector)
print("Number of dimensions of vector:", vector.ndim)
print("Shape of vector:", vector.shape)

print("Matrix")
MATRIX = torch.tensor([[7, 8], 
                       [9, 10]])
print(MATRIX)
print("Number of dimensions of MATRIX:", MATRIX.ndim)
print("Shape of MATRIX:", MATRIX.shape)

print("Tensor")
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(TENSOR)
print("Number of dimensions of TENSOR:", TENSOR.ndim)
print("Shape of TENSOR:", TENSOR.shape) 

print("Create a random tensor of size (3, 4)")
random_tensor = torch.rand(size=(3, 4))
print(random_tensor, random_tensor.dtype)

print("Create a random tensor of size (224, 224, 3)")
random_image_size_tensor = torch.rand(size=(224, 224, 3))
print(random_image_size_tensor.shape, random_image_size_tensor.ndim)

print("Create a tensor of all zeros")
zeros = torch.zeros(size=(3, 4))
print(zeros, zeros.dtype)

print("Create a tensor of all ones")
ones = torch.ones(size=(3, 4))
print(ones, ones.dtype)

print("Use torch.arange(), torch.range() is deprecated")
zero_to_ten_deprecated = torch.range(0, 10) # Note: this may return an error in the future

print("Create a range of values 0 to 10")
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)

print("Can also create a tensor of zeros similar to another tensor")
ten_zeros = torch.zeros_like(input=zero_to_ten) # will have same shape
print(ten_zeros)

print("Default datatype for tensors is float32")
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded 

print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device)

float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16) # torch.half would also work

print(float_16_tensor.dtype)

print("Create a tensor")
some_tensor = torch.rand(3, 4)

print("Find out details about it")
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}") # will default to CPU

print("Create a tensor of values and add a number to it")
tensor = torch.tensor([1, 2, 3])
print(tensor + 10)

print("Multiply it by 10")
print(tensor * 10)

print("Tensors don't change unless reassigned")
print(tensor)

print("Subtract and reassign")
tensor = tensor - 10
print(tensor)

print("Subtract and reassign")
tensor = tensor - 10
print(tensor)

print("Can also use torch functions")
torch.multiply(tensor, 10)

print("Can also use torch functions")
torch.multiply(tensor, 10)

print("Element-wise multiplication (each element multiplies its equivalent, index 0->0, 1->1, 2->2)")
print(tensor, "*", tensor)
print("Equals:", tensor * tensor)

tensor01 = torch.tensor([1, 2, 3])
tensor01.shape

print("Element-wise matrix multiplication")
print(tensor01 * tensor01)

print("Matrix multiplication")
print(torch.matmul(tensor01, tensor01))

print("Can also use the '@' symbol for matrix multiplication, though not recommended")
print(tensor01 @ tensor01)

print("Matrix multiplication by hand")
print("(avoid doing operations with for loops at all cost, they are computationally expensive)")
value = 0
for i in range(len(tensor01)):
  value += tensor01[i] * tensor01[i]
print(value)

print("CPU times: user 773 µs, sys: 0 ns, total: 773 µs")
print("Wall time: 499 µs")

print(torch.matmul(tensor01, tensor01))

print("CPU times: user 146 µs, sys: 83 µs, total: 229 µs")
print("Wall time: 171 µs")