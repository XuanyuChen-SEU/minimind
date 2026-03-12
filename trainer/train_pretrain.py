import os
import sys


__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse  # 命令行参数解析
import time  # 时间统计
import warnings  # 警告控制
import torch
import torch.distributed as dist  # 分布式训练支持
from contextlib import nullcontext  # 上下文管理器
from torch import optim  # 优化器
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载器

# from model.MokioModel import MokioMindConfig
# from dataset.lm_dataset import PretrainDataset
# from trainer.trainer_utils import (  # 训练工具函数
#     get_lr,
#     Logger,
#     is_main_process,
#     lm_checkpoint,
#     init_distributed_mode,
#     setup_seed,
#     init_model,
#     SkipBatchSampler,
# )

# 忽略警告信息，保持输出清洁
warnings.filterwarnings("ignore")


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    pass