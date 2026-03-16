"""
trainer_utils.py —— 训练通用工具

在项目中的角色：
  - 被 train_pretrain.py、train_full_sft.py 等训练脚本统一引用（get_lr、Logger、lm_checkpoint、init_model、SkipBatchSampler 等）。
  - init_model 从 model.model 导入 MokioMindForCausalLM，从 save_dir 加载 .pth；与 eval.py 的加载逻辑类似但面向训练（可 from_weight=pretrain 等）。
  - 不依赖 MokioMind 文件夹；checkpoint 保存路径由调用方传入（如 ../checkpoints）。

提供：
  - 分布式：init_distributed_mode（NCCL）、is_main_process、Logger（仅 rank0 打印）
  - 学习率：get_lr（余弦退火，step=0→lr，step=end→0.1*lr）
  - 随机性：setup_seed（Python/numpy/torch/CUDA 全设，cudnn 确定性）
  - 检查点：lm_checkpoint（保存/加载模型、优化器、epoch/step、wandb_id；支持 DDP 与 world_size 变化）
  - 模型加载：init_model（tokenizer + MokioMindForCausalLM，可选 from_weight 权重）
  - 采样器：SkipBatchSampler（包装任意 sampler，跳过前 skip_batches 个 batch，用于断点续训）
"""
import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def is_main_process():
    """当前进程是否为主进程（rank 0 或未启用分布式）。仅主进程应写日志、保存 checkpoint。"""
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """仅在主进程打印 content，避免多卡重复日志。"""
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    余弦退火学习率：lr * (0.1 + 0.45 * (1 + cos(π * current_step / total_steps)))。
    step=0 时等于 lr，step=total_steps 时约为 0.1*lr；中间平滑下降。
    """
    return (
        lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
    )  # ！修正：原公式 step=0 时 lr=1.1*lr 超出设定值，现修正为 step=0→lr, step=end→0.1*lr


def init_distributed_mode():
    """
    根据环境变量 RANK 判断是否启动 DDP。若 RANK 未设置则返回 0（单机单卡）；
    否则 init_process_group(NCCL)，设置当前进程的 CUDA 设备为 LOCAL_RANK，并返回 local_rank。
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    """固定 Python、numpy、torch、CUDA 的随机种子，并开启 cudnn 确定性、关闭 benchmark，便于复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lm_checkpoint(
    lm_config,
    weight="full_sft",
    model=None,
    optimizer=None,
    epoch=0,
    step=0,
    wandb=None,
    save_dir="checkpoints",
    **kwargs,
):
    """
    保存或加载训练检查点。
    - 保存模式（model 不为 None）：将模型 state_dict（fp16）、optimizer、epoch、step、world_size、wandb_id 及 kwargs 中带 state_dict 的对象保存到 *_resume.pth；仅模型权重另存到 *.pth（覆盖写入用 .tmp 原子替换）。
    - 加载模式（model 为 None）：若存在 _resume.pth 则加载并返回字典；若保存时的 world_size 与当前不一致，会按比例折算 step。
    kwargs 中若 value 有 state_dict()（如 scaler），会一并写入 resume 文件。
    """
    os.makedirs(save_dir, exist_ok=True)

    moe_path = "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth"
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth"

    if model is not None:
        from torch.nn.parallel import DistributedDataParallel

        # DDP 包装下需取 .module 得到真实模型再取 state_dict
        if isinstance(model, DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        ckp_tmp = ckp_path + ".tmp"
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)  # 原子覆盖，避免写入一半崩溃

        wandb_id = None
        if wandb:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                wandb_id = getattr(run, "id", None) if run else None
            else:
                wandb_id = getattr(wandb, "id", None)

        resume_data = {
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "wandb_id": wandb_id,
        }

        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "state_dict"):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + ".tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

    else:  # 加载模式：只读 _resume.pth，不读纯权重 .pth
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location="cpu")
            saved_ws = ckp_data.get("world_size", 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1

            if saved_ws != current_ws:
                ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
                Logger(
                    f"GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data['step']}"
                )

            return ckp_data
        return None


def init_model(
    lm_config,
    from_weight="pretrain",
    tokenizer_path=None,
    save_dir="../out",
    device="cuda",
):
    """
    构建 MokioMindForCausalLM 与 tokenizer。tokenizer 从 tokenizer_path 或项目 model 目录加载；
    若 from_weight 不为 "none"，从 save_dir/{from_weight}_{hidden_size}[_moe].pth 加载权重（strict=False）。
    返回 (model.to(device), tokenizer)。
    """
    from transformers import AutoTokenizer
    from model.model import MokioMindForCausalLM

    if tokenizer_path is None:
        # 获取当前文件所在目录的父目录（项目根目录）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        tokenizer_path = os.path.join(project_root, "model")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model = MokioMindForCausalLM(lm_config)

    if from_weight != "none":
        moe_suffix = (
            "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
        )
        weight_path = (
            f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
        )

        weights = torch.load(weight_path, map_location=device)

        model.load_state_dict(weights, strict=False)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f"所加载Model可训练参数：{total_params / 1e6:.3f} 百万")

    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """
    包装任意 Sampler/迭代器，按 batch_size 组成 batch，但前 skip_batches 个 batch 不产出，
    用于断点续训时跳过已训练过的 step。__len__ = max(0, 总 batch 数 - skip_batches)。
    """

    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0

        for idx in self.sampler:
            batch.append(idx)

            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue

                yield batch
                batch = []

        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
