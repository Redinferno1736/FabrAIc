import torch

# Check if CUDA (GPU) is available
print("CUDA available:", torch.cuda.is_available())

# Get number of GPUs
print("Number of GPUs:", torch.cuda.device_count())

# Get current GPU name (if available)
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
