import transformers
import torch
print("transformers version:", transformers.__version__)
print("torch version:", torch.__version__)
print("CPU threads torch can use:", torch.get_num_threads())