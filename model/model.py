"""
MokioMind 主干模型 (model.py)

在项目中的角色：
  - 被 eval.py 与 trainer 共同使用：eval 中通过 MokioMindConfig + MokioMindForCausalLM 加载 .pth；
    trainer 中通过 trainer_utils.init_model() 构造同一类模型用于训练。
  - 与 dataset/lm_dataset 的 tokenizer、vocab_size 等约定一致；与 MokioMind 文件夹中的 demo 无依赖关系。
  - 权重文件通常保存在 out/ 或 save_dir，命名如 pretrain_512.pth、full_sft_512_moe.pth。

结构概览：
  - MokioMindConfig: 配置类（hidden_size、层数、GQA、RoPE、MoE 等）
  - RMSNorm: 预/后 LayerNorm 的替代，无 bias，最后一维归一化
  - precompute_freqs / apply_rotary_pos_emb: RoPE 位置编码（支持 YaRN 外推）
  - repeat_kv: GQA 下将 KV 头复用到 Q 头数量
  - Attention: 多头自注意力（RoPE、KV cache、Flash Attention 可选）
  - FeedForward: SwiGLU 前馈（gate * up -> down）
  - MoEGate / MoEFeedForward: MoE 门控与多专家前馈
  - MokioMindBlock: 单层 = pre-norm Attention + 残差 + pre-norm FFN/MoE + 残差
  - MokioMindModel: Embedding + N × Block + 最终 RMSNorm
  - MokioMindForCausalLM: Model + lm_head（与 embed 权重共享），兼容 GenerationMixin
"""
from typing import Optional
from transformers import PretrainedConfig


class MokioMindConfig(PretrainedConfig):
    """
    MokioMind 模型配置。继承 HuggingFace PretrainedConfig，便于保存/加载。

    重要参数：
      - num_attention_heads / num_key_value_heads: GQA，KV 头数可小于 Q 头数（n_rep = heads // kv_heads）
      - rope_theta / inference_rope_scaling: RoPE 基与是否启用 YaRN 外推（如 4x）
      - use_moe: 是否用 MoE 前馈（n_routed_experts + n_shared_experts）
      - flash_attention: 是否优先使用 scaled_dot_product_attention（因果 mask）
    """
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


import torch
import math
import torch.nn as nn
from torch.nn import init
from typing import Optional, Tuple, List, Union, Any, Dict
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(nn.Module):
    """
    RMS 归一化（无 bias）：y = weight * x / sqrt(mean(x^2) + eps)。
    输入/输出形状一致：x [..., dim] -> [..., dim]，在最后一维上做 RMS。
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # [dim]

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim] -> mean(-1, keepdim=True) 得 [..., 1]，广播后 x 形状不变
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: [..., dim]；输出: [..., dim]，形状不变
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    预计算 RoPE 的 cos/sin 表，供 apply_rotary_pos_emb 使用。
    dim: 单头维度（head_dim），即每侧旋转的维度数（freqs 长度为 dim//2）。
    end: 最大位置数，决定位置维度长度。
    返回: freqs_cos, freqs_sin，形状均为 [end, dim]，与位置索引、头维度对齐。
    """
    # 1. 初始化标准 RoPE 频率。torch.arange(0, dim, 2)[:(dim//2)] 长度为 dim//2。
    # freqs 形状 [dim//2]，对应 1 / (base ** (2i/d))，i = 0,1,...,dim//2-1
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )

    if rope_scaling is not None:
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # 只有当要推断的长度大于原始训练长度时，才应用缩放
        if end / orig_max > 1.0:
            # 3. 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )

            # 4. 计算高频区和低频区的维度切分点
            # low: 不需要缩放的高频部分的最高索引
            # high: 需要完全缩放的低频部分的最低索引
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # 5. 计算混合因子 γ (Ramp)
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
            freqs = freqs * (1 - ramp + ramp / factor)

    # 7. 根据目标长度 end，生成位置索引向量 t
    t = torch.arange(end, device=freqs.device)  # [end]

    # 8. 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    freqs = torch.outer(t, freqs).float()  # [end, dim//2]

    # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)；沿 dim=-1 复制一份得到完整 head_dim
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor  # [end, dim]
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor  # [end, dim]

    return freqs_cos, freqs_sin  # 均为 [end, dim]，dim = head_dim


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 Q、K 应用旋转位置编码（RoPE）。cos/sin 形状 [seq, head_dim]，会在 unsqueeze_dim 上扩维以广播。
    公式：q_embed = q*cos + rotate_half(q)*sin（对 k 同理）。rotate_half 为后半维取反后与前半维拼接。
    输入 q: [bsz, seq, n_heads, head_dim], k: [bsz, seq, n_kv_heads, head_dim]；
    cos/sin: [seq, head_dim]，unsqueeze(1) 后 [1, seq, 1, head_dim]，与 q/k 广播。
    输出 q_embed/k_embed: 与 q/k 形状相同。
    """
    def rotate_half(x):
        # x: [..., head_dim] -> 后半维取反与前半维拼接，仍 [..., head_dim]
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    # cos/sin unsqueeze 后与 q/k 广播，逐元素运算，输出形状同 q/k
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    GQA：将 KV 头沿头维度重复 n_rep 次，使 KV 头数与 Q 头数一致，便于做 attention。
    输入 x: [bsz, slen, num_kv_heads, head_dim]；
    中间 expand 后: [bsz, slen, num_kv_heads, n_rep, head_dim]；
    输出: [bsz, slen, num_kv_heads * n_rep, head_dim]（与 Q 头数一致）。
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :]  # [bsz, slen, num_kv_heads, 1, head_dim]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    多头自注意力（支持 GQA、RoPE、KV cache、Flash Attention）。
    前向：x [bsz, seq_len, hidden_size] -> Q/K/V 投影 -> RoPE -> (可选) 拼 past_kv -> repeat_kv(K,V)
    -> attention(scores 因果 mask + 可选 padding mask) -> 输出投影。
    返回 output [bsz, seq_len, hidden_size], past_kv = (K, V) 或 None；
    past 中 K/V 形状均为 [bsz, n_kv_heads, total_len, head_dim]，total_len = past_len + cur_seq_len。
    """

    def __init__(self, args: MokioMindConfig):
        super().__init__()

        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )

        assert args.num_attention_heads % self.num_key_value_heads == 0, f"num_attention_heads {args.num_attention_heads} must be divisible by num_key_value_heads {self.num_key_value_heads}"

        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # 输入 x: [bsz, seq_len, hidden_size]
        bsz, seq_len, _ = x.shape

        # 线性投影（最后一维变化，前两维不变）后按头 reshape：
        # q_proj(x) -> [bsz, seq_len, n_heads*head_dim] -> view -> [bsz, seq_len, n_local_heads, head_dim]
        # k_proj/v_proj(x) -> [bsz, seq_len, n_kv_heads*head_dim] -> view -> [bsz, seq_len, n_local_kv_heads, head_dim]
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # RoPE：cos/sin 形状 [cur_len, head_dim]，cur_len = seq_len（无 cache）或与 position 对应长度
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        # xq 仍 [bsz, seq_len, n_local_heads, head_dim]，xk/xv 仍 [bsz, seq_len, n_local_kv_heads, head_dim]

        # KV cache 输入：past_key_value = (K, V)，K/V 形状均为 [bsz, past_len, n_kv_heads, head_dim]；在 dim=1(序列维) 上拼当前步
        # 拼接后 xk/xv: [bsz, past_len+seq_len, n_local_kv_heads, head_dim]；若 use_cache 则返回的 past_kv 中 K/V 即此形状
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 转为 [bsz, n_heads, kv_len, head_dim] 以做 batch matmul；repeat_kv 使 KV 头数 = n_local_heads
        # xq: [bsz, seq_len, n_heads, head_dim] -> transpose(1,2) -> [bsz, n_heads, seq_len, head_dim]
        # xk/xv 经 repeat_kv 后 [bsz, kv_len, n_heads, head_dim] -> transpose -> [bsz, n_heads, kv_len, head_dim]，kv_len = past_len+seq_len
        xq, xk, xv = (
            xq.transpose(1, 2),  # [bsz, n_local_heads, seq_len, head_dim]
            repeat_kv(xk, self.n_rep).transpose(1, 2),  # [bsz, n_local_heads, kv_len, head_dim]
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        # Flash Attention 路径：无 padding、无 past 时用因果 SDPA；否则手写 scores + 因果 mask + padding mask
        if (
            self.flash
            and (seq_len > 1)
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            # SDPA 输入/输出：q [bsz, n_heads, seq_len, head_dim], k/v [bsz, n_heads, seq_len, head_dim]
            # 输出 output: [bsz, n_heads, seq_len, head_dim]
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # xq @ xk^T: [bsz, n_heads, seq_len, head_dim] @ [bsz, n_heads, head_dim, kv_len] -> [bsz, n_heads, seq_len, kv_len]
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 仅对“当前步 q 对当前步 k”子块加因果 mask：[seq_len, seq_len] 上三角为 -inf
            scores[:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1,
            )

            if attention_mask is not None:
                # attention_mask: [bsz, kv_len]，1=有效 0=pad；扩维成 [bsz, 1, 1, kv_len] 与 scores 广播
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # [bsz, n_heads, seq_len, kv_len]
            scores = self.attn_dropout(scores)
            output = scores @ xv  # [bsz, n_heads, seq_len, kv_len] @ [bsz, n_heads, kv_len, head_dim] -> [bsz, n_heads, seq_len, head_dim]

        # 合并头维度：transpose(1,2) -> [bsz, seq_len, n_heads, head_dim]，reshape -> [bsz, seq_len, n_heads*head_dim]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))  # [bsz, seq_len, hidden_size]
        return output, past_kv


class FeedForward(nn.Module):
    """
    SwiGLU 前馈：out = down( act(gate(x)) * up(x) )。
    形状：x [..., hidden_size] -> gate/up -> [..., intermediate_size] -> 逐元乘 -> down -> [..., hidden_size]。
    intermediate_size 默认约为 hidden_size * 8/3，并向上取整到 64 的倍数。
    """

    def __init__(self, config: MokioMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: [bsz, seq_len, hidden_size]（或任意 [..., hidden_size]）
        # gate_proj(x): [..., intermediate_size], up_proj(x): [..., intermediate_size]
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)  # [..., intermediate_size]
        return self.dropout(self.down_proj(gated))  # [..., hidden_size]，与输入 x 前两维一致


class MoEGate(nn.Module):
    """
    MoE 门控：对每个 token 的 hidden 做线性变换得到 n_routed_experts 维 logits，
    经 softmax 后取 top_k 专家及其权重；可选对 top_k 权重再归一化（norm_topk_prob）。
    返回 topk_idx [bsz*seq, top_k], topk_weight [bsz*seq, top_k], aux_loss（标量，负载均衡辅助损失）。
    """

    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))  # [n_routed_experts, hidden_size]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 输入 hidden_states: [bsz, seq_len, hidden_size]
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)  # [bsz*seq_len, hidden_size]
        logits = F.linear(hidden_states, self.weight, None)  # [bsz*seq_len, n_routed_experts]

        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)  # [bsz*seq_len, n_routed_experts]
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        # topk_weight: [bsz*seq_len, top_k], topk_idx: [bsz*seq_len, top_k]，每个元素为专家 id (0..n_routed_experts-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator  # 每个 token 的 top_k 权重和为 1

        # ---------- 负载均衡辅助损失（仅训练时、alpha>0 时计算）----------
        # 目的：鼓励各专家被均匀使用，避免部分专家过载、部分闲置。与主 loss 加权相加后反向传播。
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores  # [bsz*seq_len, n_routed_experts]，门控 softmax 分数
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)  # [bsz, seq_len*top_k]，每个位置选中的专家 id

            # 序列级辅助损失（seq_aux=True）：按 batch 内每个样本单独统计专家使用率，再与门控分数结合
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)  # [bsz, seq_len, n_routed_experts]
                # ce[b,i] = 样本 b 中专家 i 被选中的次数，再除以 (seq_len*top_k/n_routed_experts)，使理想均匀时 ce 全为 1
                ce = torch.zeros(  # [bsz, n_routed_experts]
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,  # 列索引：专家 id
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),  # 每次选中加 1
                ).div_(seq_len * aux_topk / self.n_routed_experts)  # 归一化：均匀时 ce[b,i]=1
                # 每个样本的损失：sum_i( ce[b,i] * mean_over_seq(scores[b,:,i]) )，再对 b 取平均，乘 alpha
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha  # 标量
            # 样本级辅助损失（seq_aux=False）：在整个 batch 上统计专家使用频率，与门控平均分数结合
            else:
                # one-hot：每个 (batch,seq,top_k) 位置选中的专家记为 1
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )  # [bsz*seq_len*top_k, n_routed_experts]
                ce = mask_ce.float().mean(0)  # [n_routed_experts]，专家 i 被选中的全局比例 f_i
                Pi = scores_for_aux.mean(0)  # [n_routed_experts]，门控对专家 i 的平均分数 P_i
                fi = ce * self.n_routed_experts  # 均匀时 f_i=1/n，故 f_i*N=1；负载高时 >1
                # 辅助项：sum_i P_i * f_i * N，鼓励 P 与 f 一致（负载均衡）
                aux_loss = (Pi * fi).sum() * self.alpha  # 标量
        else:
            aux_loss = scores.new_zeros(1).squeeze()  # 标量 0，推理或不启用辅助损失时
        return topk_idx, topk_weight, aux_loss


class MoEFeedForward(nn.Module):  # ！修正：原MoEFeedForaward拼写错误
    """
    MoE 前馈：每个 token 由 gate 选出 top_k 个专家，经专家 FFN 后按权重加权求和；
    若有 n_shared_experts，再加一次 shared 专家（对所有 token 相同）。
    训练：按 token 遍历专家做前向再加权；推理：moe_infer 按专家打包，减少 kernel 调用。
    会设置 self.aux_loss 供上层汇总（负载均衡损失）。
    """

    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        # 专家层：n_routed_experts 个独立 FeedForward
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.n_routed_experts)]
        )
        # 门控层：输出 top_k 专家索引与权重
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: [bsz, seq_len, hidden_size]
        identity = x
        orig_shape = x.shape
        bsz, seq_len, h = orig_shape

        # 门控：topk_idx [bsz*seq_len, top_k], topk_weight [bsz*seq_len, top_k], aux_loss 标量
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])  # [bsz*seq_len, hidden_size]

        flat_topk_idx = topk_idx.view(-1)  # [bsz*seq_len*top_k]，每个元素为专家 id (0..n_routed_experts-1)
        if self.training:
            # 训练：每个 token 复制 top_k 份，送入对应专家；x 变为 [bsz*seq_len*top_k, hidden_size]
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)  # [bsz*seq_len*top_k, hidden_size]
            for i, expert in enumerate(self.experts):
                expert_out = expert(x[flat_topk_idx == i])  # [该专家 token 数, hidden_size]
                if expert_out.shape[0] > 0:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(
                        p.sum() for p in expert.parameters()
                    )
            # y: [bsz*seq_len*top_k, hidden] -> view [bsz*seq_len, top_k, hidden] * topk_weight [bsz*seq_len, top_k, 1] -> sum(dim=1) -> [bsz*seq_len, hidden]
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)  # [bsz, seq_len, hidden_size]
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *orig_shape
            )  # moe_infer 输出 [bsz*seq_len, hidden_size]
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)  # identity: [bsz, seq_len, hidden_size]
        self.aux_loss = aux_loss
        return y  # [bsz, seq_len, hidden_size]

    @torch.no_grad()
    def moe_infer(
        self,
        x: torch.Tensor,
        flat_expert_indices: torch.Tensor,
        flat_expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        MoE 推理：按专家打包，每个专家一次性处理分到它的所有 token，再按原始 token 顺序 scatter 回结果。
        输入：
          x: [bsz*seq_len, hidden_size]
          flat_expert_indices: [bsz*seq_len*top_k]，每个位置对应选中的专家 id
          flat_expert_weights: [bsz*seq_len*top_k, 1] 或可广播形状，对应权重
        输出：expert_cache [bsz*seq_len, hidden_size]，与 x 同形状。
        流程：argsort 将 (token_idx, expert_id) 按专家 id 排序 -> 按专家切段 -> 每段批量过对应 expert -> 加权 -> scatter_add 回 expert_cache。
        """
        expert_cache = torch.zeros_like(x)  # [bsz*seq_len, hidden_size]
        idxs = flat_expert_indices.argsort()  # [bsz*seq_len*top_k]，排序后同一专家的条目连续
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)  # 每个专家的累计 token 数
        token_idxs = idxs // self.config.num_experts_per_tok  # [bsz*seq_len*top_k]，排序后位置对应的原始 token 下标
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]  # 当前专家负责的 token 下标
            expert_tokens = x[exp_token_idx]  # [当前专家 token 数, hidden_size]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)  # [当前专家 token 数, hidden_size]
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])  # 逐 token 乘权重
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )  # 按原始 token 下标加回 expert_cache

        return expert_cache  # [bsz*seq_len, hidden_size]


class MokioMindBlock(nn.Module):
    """
    单层 Transformer 块：pre-norm Attention + 残差 + pre-norm FFN/MoE + 残差。
    输入/输出 hidden_states 形状均为 [bsz, seq_len, hidden_size]；另返回 present_key_value 供 KV cache。
    """

    def __init__(self, layer_id: int, config: MokioMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attention = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = (
            FeedForward(config)
            if not config.use_moe
            else MoEFeedForward(config)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # 输入 hidden_states: [bsz, seq_len, hidden_size]；position_embeddings: (cos, sin) 各 [cur_len, head_dim]
        # past_key_value: 单层 KV cache，(K, V) 或 None；K/V 形状 [bsz, past_len, n_kv_heads, head_dim]
        res = hidden_states

        # pre-norm: RMSNorm 不改变形状；Attention 输入/输出均为 [bsz, seq_len, hidden_size]
        hidden_states, present_key_value = self.self_attention(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )

        hidden_states = res + hidden_states  # 残差 [bsz, seq_len, hidden_size]

        # pre-norm + MLP/MoE + 残差；mlp 输入/输出 [bsz, seq_len, hidden_size]
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value  # present_key_value: (K, V) 或 None；K/V 形状 [bsz, total_len, n_kv_heads, head_dim]，total_len=past_len+seq_len


class MokioMindModel(nn.Module):
    """
    主干：Embedding -> N × MokioMindBlock -> 最终 RMSNorm。
    前向：input_ids [bsz, seq_len] -> hidden [bsz, seq_len, hidden] -> 每层用 start_pos:start_pos+seq_len 的 RoPE 切片。
    返回 hidden_states [bsz, seq_len, hidden], presents (每层 KV), aux_loss（MoE 负载均衡和）。
    """

    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [MokioMindBlock(l, config) for l in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, List[Optional[Tuple[torch.Tensor, torch.Tensor]]], torch.Tensor]:
        # 输入 input_ids: [bsz, seq_len]，attention_mask: [bsz, seq_len] 或 [bsz, total_len]（1=有效，0=pad）
        # past_key_values: 长度为 num_hidden_layers 的列表，第 l 项为 (K_l, V_l) 或 None；
        #   每层 K/V 形状为 [bsz, past_len, n_kv_heads, head_dim]，past_len 为已缓存的序列长度（首次为 0）
        batch_size, seq_length = input_ids.shape

        if hasattr(past_key_values, "layers"):
            past_key_values = None

        past_key_values = past_key_values or [None] * len(self.layers)

        # start_pos：已有 KV 的序列长度；past_key_values[l][0] 为 K，形状 [bsz, past_len, n_kv_heads, head_dim]，取 .shape[1] 得 past_len
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        # Embedding：input_ids [bsz, seq_len] -> embed_tokens -> [bsz, seq_len, hidden_size]；dropout 不改变形状
        hidden_states = self.dropout(
            self.embed_tokens(input_ids)
        )

        # 当前步 RoPE 片段：从预计算的 freqs_cos/freqs_sin 中切片，各 [seq_length, head_dim]
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length],
        )
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)  # present: (K, V) 或 None；K/V 形状 [bsz, total_len, n_kv_heads, head_dim]，total_len=past_len+seq_len

        hidden_states = self.norm(hidden_states)  # [bsz, seq_len, hidden_size]

        # MoE 负载均衡辅助损失：仅当使用 MoEFeedForward 的层才有 aux_loss（标量）
        aux_loss = sum(
            [
                layer.mlp.aux_loss
                for layer in self.layers
                if isinstance(
                    layer.mlp, MoEFeedForward
                )  # ！修正：原MoEFeedForaward拼写错误
            ],
            hidden_states.new_zeros(1).squeeze(),
        )

        return hidden_states, presents, aux_loss  # presents: 长度为 num_hidden_layers，每项 (K,V) 中 K/V 为 [bsz, total_len, n_kv_heads, head_dim]


class MokioMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    因果语言模型：MokioMindModel + lm_head；embed_tokens 与 lm_head 权重共享（同一 Linear weight）。
    前向：input_ids [bsz, seq_len] -> model -> hidden_states [bsz, seq_len, hidden_size]
         -> lm_head(hidden_states[:, slice, :]) -> logits [bsz, kept, vocab_size]。
    logits_to_keep：生成时通常为 1（只保留最后一步），训练时可保留整段；labels [bsz, seq_len] 存在时计算 CE loss（shift 后、ignore_index=-100）。
    """
    config_class = MokioMindConfig

    def __init__(self, config: MokioMindConfig):
        super().__init__(config)
        self.model = MokioMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight  # 权重共享

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **args: Any,
    ) -> CausalLMOutputWithPast:
        # past_key_values 入参/出参形状：List of length num_hidden_layers，每层 (K, V)，K/V 为 [bsz, past_len或total_len, n_kv_heads, head_dim]
        # model 返回 hidden_states [bsz, seq_len, hidden_size], past_key_values（同上形状）, aux_loss
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )

        # 只对最后 logits_to_keep 个位置做 lm_head（生成时多为 1），减少计算
        # hidden_states[:, slice_indices, :]: [bsz, kept, hidden_size]
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])  # [bsz, kept, vocab_size]

        loss = None
        if labels is not None:
            # 自回归：预测下一个 token；labels [bsz, seq] 与 input 对齐，shift 后与 logits 对齐
            # shift_logits: [bsz, kept-1, vocab_size]，shift_labels: [bsz, kept-1]（或与 logits 第二维一致）
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),  # [bsz*(kept-1), vocab_size]
                shift_labels.view(-1),  # [bsz*(kept-1)]
                ignore_index=-100,
            )

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
        output.aux_loss = aux_loss
        return output
