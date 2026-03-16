"""
test_cuda.py —— 环境检查脚本

用途：快速检查当前环境是否可用 CUDA 以及 PyTorch 版本。
运行：python test_cuda.py
  - torch.cuda.is_available() 为 True 时，训练/推理脚本会使用 GPU。
  - 若为 False，需检查驱动、CUDA 与 PyTorch 的 GPU 版本安装。
"""
import torch

print(torch.cuda.is_available())
print(torch.__version__)