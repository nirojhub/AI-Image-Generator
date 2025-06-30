import torch
from datetime import datetime
print(f"{torch.cuda.is_available()}")
print(f"{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}")