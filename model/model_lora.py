"""
LoRA（Low-Rank Adaptation）适配模块 (model_lora.py)

在项目中的角色：
  - 可选功能：eval.py 中若传入 --lora_weight 且取消顶部 apply_lora/load_lora 的注释，会在加载主权重后再加载 LoRA。
  - 训练脚本（train_full_sft 等）也可在加载 base 模型后调用 apply_lora，仅训练 LoRA 参数。
  - 依赖 model.model 中的 MokioMindForCausalLM（或其它 nn.Module），不依赖 MokioMind 文件夹。

用途：在冻结原模型的前提下，仅训练低秩矩阵 A、B，使输出变为 Wx + B(Ax)，节省显存与训练量。
  - LoRA: 单层低秩模块，y = B(A(x))；A [in, rank]，B [rank, out]，初始时 B=0 故等价于未加 LoRA。
  - apply_lora: 对所有「方阵」Linear（in_features == out_features）挂载 LoRA，forward 改为 original(x) + lora(x)。
  - load_lora / save_lora: 按子模块路径加载/保存各 module.lora 的 state_dict，兼容 "module." 前缀（DDP）。
"""
import torch
from torch import optim, nn


class LoRA(nn.Module):
    """
    低秩适配：out = B(A(x))，等价于在原始线性层上加上低秩增量 W' = B @ A。
    形状：x [..., in_features] -> A -> [..., rank] -> B -> [..., out_features]。
    A 高斯初始化、B 零初始化，使训练初期的输出与未加 LoRA 时一致。
    """

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=8):
    """
    为模型中所有「方阵」Linear 层挂载 LoRA 子模块，并替换 forward 为 原层(x) + lora(x)。
    仅当 weight.shape[0] == weight.shape[1] 时挂载（避免误改 embedding 或 lm_head 等非方线性层）。
    """
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and module.weight.shape[0] == module.weight.shape[1]
        ):
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(
                device
            )
            setattr(module, "lora", lora)
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


def load_lora(model, path):
    """
    从 path 加载 LoRA 权重到已 apply_lora 的 model。
    会去掉 state_dict key 的 "module." 前缀（DDP 保存的格式），再按 "name.lora." 匹配到各 module.lora。
    """
    device = next(model.parameters()).device
    state_dict = torch.load(path, map_location=device)
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }

    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state = {
                k.replace(f"{name}.lora.", ""): v
                for k, v in state_dict.items()
                if f"{name}.lora." in k
            }
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """
    将模型中所有 module.lora 的 state_dict 合并保存到 path。
    若 model 被 DDP 包装，会尝试取 _orig_mod；key 会带上 "name.lora."，与 load_lora 对应。
    """
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {
                f"{clean_name}.lora.{k}": v for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)
    torch.save(state_dict, path)
