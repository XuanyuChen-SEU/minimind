"""
train_pretrain.py —— MokioMind 自回归预训练脚本

在项目中的角色：
  - 训练流水线第一步：通常 from_weight=none 从头训练，或加载既有 pretrain 权重继续预训练。
  - 使用 dataset.lm_dataset.PretrainDataset（数据格式 {"text": "..."}）、model.model.MokioMindConfig、trainer_utils 的 init_model/lm_checkpoint 等。
  - 运行方式：python -m trainer.train_pretrain [--data_path ../dataset/xxx.jsonl] [--from_weight none]；输出权重可被 train_full_sft 的 --from_weight pretrain 使用。

流程概览：
  1. 解析参数（保存路径、epochs、batch_size、学习率、混合精度、模型尺寸、数据路径、from_weight/from_resume 等）
  2. init_distributed_mode / setup_seed
  3. 构建 MokioMindConfig，可选从 lm_checkpoint 加载断点（resume）
  4. 设置 autocast（bfloat16/float16）、可选 wandb
  5. init_model、PretrainDataset、DistributedSampler、GradScaler、AdamW；若有 ckp_data 恢复模型/优化器/scaler 与 start_epoch/start_step
  6. 可选 DDP 包装（忽略 freqs_cos/freqs_sin）
  7. 每 epoch：设置 sampler epoch；若断点续训则用 SkipBatchSampler 跳过前 start_step 个 batch；DataLoader + train_epoch

训练逻辑：PretrainDataset 返回 (input_ids, labels, attention_mask)；前向时传入 labels 做 CE loss（模型内部 shift），梯度累积后 clip_grad_norm、scaler.step、按间隔保存与打日志。
"""
from __future__ import annotations

import os
import sys


__package__ = "trainer"
# 将项目目录导入到 sys.path，以便导入项目中的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from typing import Any, Optional, Iterable, Tuple

from model.model import MokioMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)

warnings.filterwarnings("ignore")


def train_epoch(
    epoch: int,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    iters: int,
    start_step: int = 0,
    wandb: Optional[Any] = None,
) -> None:
    """
    执行一个 epoch 的预训练。loader 每步产出 (input_ids, labels, attention_mask)，形状 [batch_size, max_seq_len]。
    学习率按 get_lr 余弦退火；loss 做梯度累积后 clip、scaler.step；按 log_interval 打日志、save_interval 保存 checkpoint。
    """
    start_time = time.time()

    for step, (input_ids, labels, attention_mask) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        labels = labels.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:  # 混合精度上下文（自动用 FP16 计算，FP32 存梯度）
            res = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = res.loss  # 预训练损失（如 CE 损失）
            loss = loss / args.accumulation_steps  # 梯度累积：损失均分

        scaler.scale(loss).backward()  # 缩放损失，避免 FP16 下溢


        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 反缩放梯度（为了梯度裁剪）
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪
            
            scaler.step(optimizer)  # 更新优化器（自动处理混合精度）
            scaler.update()  # 更新缩放器状态

            optimizer.zero_grad(set_to_none=True)  # 清空梯度（set_to_none=True 更省显存）

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps # 还原真实损失（因为之前除以了累积步数）
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:"
            )

            # 记录到实验跟踪系统
            if wandb:
                wandb.log(
                    {"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min}
                )

            if (step % args.save_interval == 0 or step == iters) and is_main_process():
                model.eval()  # 切换到评估模式（避免 Dropout/BatchNorm 影响权重）

                # 生成 MoE 模型的文件名后缀
                moe_suffix = "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
                ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"

                # 处理 DDP 模型，获取真实 state_dict（去掉 module. 前缀）
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                state_dict = {k: v.half() for k, v in state_dict.items()}  # 转为 FP16，减小文件体积
                torch.save(state_dict, ckp)  # 保存纯模型权重（用于推理/部署）

                # 保存完整检查点（用于断点续训）
                lm_checkpoint(
                    lm_config,
                    weight=args.save_weight,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    step=step,
                    wandb=wandb,
                    save_dir="../checkpoints",
                )

                model.train()  # 切回训练模式


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MokioMind Pretraining")

    # ========== 基础训练参数 ==========
    parser.add_argument(
        "--save_dir", type=str, default="../out", help="模型保存目录"
    )
    parser.add_argument(
        "--save_weight", default="pretrain", type=str, help="保存权重的前缀名"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")

    # ========== 硬件和性能参数 ==========
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")

    # ========== 训练策略参数 ==========
    parser.add_argument(
        "--accumulation_steps", type=int, default=8, help="梯度累积步数"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")

    # ========== 模型架构参数 ==========
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--max_seq_len", default=512, type=int, help="训练的最大截断长度"
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )

    # ========== 数据和恢复参数 ==========
    parser.add_argument(
        "--data_path",
        type=str,
        default="../dataset/pretrain_hq.jsonl",  # ！修正：原"dataset/..."缺少../前缀
        help="预训练数据路径",
    )
    parser.add_argument(
        "--from_weight",
        default="none",
        type=str,
        help="基于哪个权重训练，为none则从头开始",
    )
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测&续训（0=否，1=是）",
    )

    # ========== 实验跟踪参数 ==========
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument(
        "--wandb_project", type=str, default="MokioMind-Pretrain", help="wandb项目名"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # ---------- 1. 分布式与随机种子 ----------
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"

    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ---------- 2. 目录、配置、断点 ----------
    # lm_config 的 hidden_size / num_hidden_layers / use_moe 需与数据集、后续 SFT 及 eval 使用的模型一致
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = MokioMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    ckp_data = (
        lm_checkpoint(
            lm_config, weight=args.save_weight, save_dir="../checkpoints"
        )
        if args.from_resume == 1
        else None
    )

    # ---------- 3. 混合精度 ----------
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ---------- 4. 实验跟踪（SwanLab 作 wandb 接口） ----------
    wandb = None
    if args.use_wandb and is_main_process():
        # 使用SwanLab作为WandB的替代
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"MokioMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    # ---------- 5. 模型、数据、优化器、断点恢复 ----------
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), args.batch_size, start_step
            )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
            )
            train_epoch(epoch, loader, len(loader) + start_step, start_step, wandb)
        else:
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            train_epoch(epoch, loader, len(loader), 0, wandb)
