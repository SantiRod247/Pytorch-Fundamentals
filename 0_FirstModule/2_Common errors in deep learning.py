import torch

print("View tensor_A and tensor_B")
tensor_A = torch.tensor([[1., 2.],
                        [3., 4.],
                        [5., 6.]])
tensor_B = torch.tensor([[ 7., 10.],
                        [ 8., 11.],
                        [ 9., 12.]])

print(tensor_A)
print(tensor_B)

print("View tensor_A and tensor_B.T")
print(tensor_A)
print(tensor_B.T)

print("The operation works when tensor_B is transposed")
print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}\n")
print(f"New shapes: tensor_A = {tensor_A.shape} (same as above), tensor_B.T = {tensor_B.T.shape}\n")
print(f"Multiplying: {tensor_A.shape} * {tensor_B.T.shape} <- inner dimensions match\n")
print("Output:\n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output) 
print(f"\nOutput shape: {output.shape}")

print("torch.mm is a shortcut for matmul")
print(torch.mm(tensor_A, tensor_B.T))

print("Since the linear layer starts with a random weights matrix, let's make it reproducible (more on this later)")
torch.manual_seed(42)
print("This uses matrix multiplication")
linear = torch.nn.Linear(in_features=2, # in_features = matches inner dimension of input 
                         out_features=6) # out_features = describes outer value 
x = tensor_A
output = linear(x)
print(f"Input shape: {x.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}")


print("Create a tensor")
x = torch.arange(0, 100, 10)
print(x)

print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
print("print(f\"Mean: {x.mean()}\") # this will error")
print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype
print(f"Sum: {x.sum()}")

print(torch.max(x), torch.min(x), torch.mean(x.type(torch.float32)), torch.sum(x))

print("Create a tensor")
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")

print("Returns index of max and min values")
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")

print("Create a tensor and check its datatype")
tensor = torch.arange(10., 100., 10.)
print(tensor.dtype)

print("Create a float16 tensor")
tensor_float16 = tensor.type(torch.float16)
print(tensor_float16)

print("Create an int8 tensor")
tensor_int8 = tensor.type(torch.int8)
print(tensor_int8)

print("Method\tOne-line description")
print("torch.reshape(input, shape)\tReshapes input to shape (if compatible), can also use torch.Tensor.reshape().")
print("Tensor.view(shape)\tReturns a view of the original tensor in a different shape but shares the same data as the original tensor.")
print("torch.stack(tensors, dim=0)\tConcatenates a sequence of tensors along a new dimension (dim), all tensors must be same size.")
print("torch.squeeze(input)\tSqueezes input to remove all the dimenions with value 1.")
print("torch.unsqueeze(input, dim)\tReturns input with a dimension value of 1 added at dim.")
print("torch.permute(input, dims)\tReturns a view of the original input with its dimensions permuted (rearranged) to dims.")

x = torch.arange(1., 8.)
print(x, x.shape)

print("Add an extra dimension")
x_reshaped = x.reshape(1, 7)
print(x_reshaped, x_reshaped.shape)

print("Change view (keeps same data as original but changes view)")
print("See more: https://stackoverflow.com/a/54507446/7900723")
z = x.view(1, 7)
print(z, z.shape)

print("Changing z changes x")
z[:, 0] = 5
print(z, x)

print("Stack tensors on top of each other")
x_stacked = torch.stack([x, x, x, x], dim=0) # try changing dim to dim=1 and see what happens
print(x_stacked)

print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

print("Remove extra dimension from x_reshaped")
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")

print(f"Previous tensor: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")

print("Add an extra dimension with unsqueeze")
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")

print("Create tensor with specific shape")
x_original = torch.rand(size=(224, 224, 3))

print("Permute the original tensor to rearrange the axis order")
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0

print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}")