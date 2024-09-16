import torch 
print(torch.__version__)
print(torch.cuda.is_available())
print("If the above outputs True, PyTorch can see and use the GPU, if it outputs False, it can't see the GPU.")

print("Set device type")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

print("If the above output \"cuda\" it means we can set all of our PyTorch code to use the available CUDA device (a GPU) and if it output \"cpu\", our PyTorch code will stick with the CPU.")

print("Count number of devices")
print(torch.cuda.device_count())

print("Create tensor (default on CPU)")
tensor = torch.tensor([1, 2, 3])

print("Tensor not on GPU")
print(tensor, tensor.device)

print("Move tensor to GPU (if available)")
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)

print("Notice the second tensor has device='cuda:0', this means it's stored on the 0th GPU available (GPUs are 0 indexed, if two GPUs were available, they'd be 'cuda:0' and 'cuda:1' respectively, up to 'cuda:n').")  

print("If tensor is on GPU, can't transform it to NumPy (this will error)")
print("# tensor_on_gpu.numpy()")

print("Instead, copy the tensor back to cpu")
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
print(tensor_back_on_cpu)
print(tensor_on_gpu)
