# RoPE 与 YaRN

本文档介绍 **RoPE（Rotary Position Embedding）** 以及其长上下文扩展方法 **YaRN**，并说明本项目中 `precompute_freqs_cls` 与 `apply_rotary_pos_emb` 的实现。

---

## 一、RoPE 概述

### 1.1 什么是 RoPE

**RoPE**（Rotary Position Embedding，旋转位置编码）是一种将位置信息编码进 attention 的 query 和 key 中的方法。与在 embedding 上直接加位置编码不同，RoPE 通过**旋转变换**把位置信息注入 Q、K 的表示，使注意力分数天然包含相对位置信息。

**参考文献**：Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021.

### 1.2 核心思想

对每个维度对 (2i, 2i+1)，将向量视为 2D 平面上的点，根据位置 `pos` 旋转角度 θ<sub>i</sub> × pos：

- 角频率：θ<sub>i</sub> = base<sup>-2i/d</sup>，其中 `base` 为 rope_base（如 1e6）
- 波长：λ<sub>i</sub> = 2π / θ<sub>i</sub>，低维（小 i）对应短波长（高频），高维对应长波长（低频）

旋转公式（2D）：
$$
\begin{bmatrix} q'_0 \\ q'_1 \end{bmatrix} = \begin{bmatrix} \cos(\theta \cdot m) & -\sin(\theta \cdot m) \\ \sin(\theta \cdot m) & \cos(\theta \cdot m) \end{bmatrix} \begin{bmatrix} q_0 \\ q_1 \end{bmatrix}
$$

即：`x_rotated = x * cos - rotate_half(x) * sin`（或等价地 `x * cos + rotate_half(x) * sin` 取决于实现约定）。

### 1.3 优点

- **相对位置感知**：注意力分数仅依赖相对位置 m−n
- **外推性**：训练长度内的模式可在更长序列上延续
- **不增加参数量**：纯几何变换，无额外可学习参数

---

## 二、YaRN 概述

### 2.1 动机

标准 RoPE 在**推理序列长度超过训练长度**时，会出现注意力分布过尖、困惑度上升等问题。需要在不大幅重训的前提下，扩展可处理的上下文长度。

### 2.2 什么是 YaRN

**YaRN**（Yet another RoPE extension method）结合 NTK-by-Parts 插值与注意力缩放，对 RoPE 进行「分频段」处理：

- **高频维度**（短波长）：保持原频率，用于局部细节
- **低频维度**（长波长）：对频率做插值缩放，拉长有效波长，从而支持更长序列
- **过渡区**：用 ramp 做平滑插值
- **注意力缩放**：用 `attn_factor` 缓解长序列下注意力分布过尖

**参考文献**：Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models*, 2023.

### 2.3 核心参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `original_max_position_embeddings` | 训练时的最大位置长度 | 2048 |
| `factor` | 扩展倍数 | 16 |
| `beta_fast` | 高频边界（不缩放） | 32 |
| `beta_slow` | 低频边界（需缩放） | 1 |
| `attention_factor` | 注意力 logits 缩放 | 1.0 |

---

## 三、`precompute_freqs_cls` 函数

用于预计算 RoPE（及 YaRN）所需的 cos/sin 频率表。定义位置：`model/model.py` 第 93–146 行。

### 3.1 函数签名

```python
def precompute_freqs_cls(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None
) -> Tuple[Tensor, Tensor]
```

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `dim` | int | - | 头维度（head_dim） |
| `end` | int | 32768 | 最大位置索引（支持的最大序列长度） |
| `rope_base` | float | 1e6 | RoPE 的 θ 基值 |
| `rope_scaling` | dict \| None | None | YaRN 参数，None 时使用标准 RoPE |

**返回值**：`(freqs_cos, freqs_sin)`，形状均为 `[end, dim]`。

### 3.2 执行流程

#### 步骤 1：初始化标准 RoPE 频率

```python
freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
attn_factor = 1.0
```

- θ<sub>i</sub> = rope_base<sup>-2i/d</sup>，i ∈ {0, 1, …, d/2−1}
- `freqs` 长度为 `dim // 2`，成对使用 cos/sin

#### 步骤 2：YaRN 缩放（当 `rope_scaling` 非空且 `end > orig_max`）

**2.1 读取参数**

```python
orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
factor = rope_scaling.get("factor", 16)
beta_fast = rope_scaling.get("beta_fast", 32.0)
beta_slow = rope_scaling.get("beta_slow", 1.0)
attn_factor = rope_scaling.get("attention_factor", 1.0)
```

**2.2 波长到维度索引**

```python
inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
low = max(math.floor(inv_dim(beta_fast)), 0)
high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
```

- `low`：索引 < low 为高频，不缩放
- `high`：索引 > high 为低频，需要缩放
- `low ~ high`：过渡区

**2.3 Ramp 与频率缩放**

```python
ramp = torch.clamp(
    (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
    0, 1
)
freqs = freqs * (1 - ramp + ramp / factor)
```

- ramp=0：系数 1，频率不变
- ramp=1：系数 1/factor，频率缩小 factor 倍
- 0<ramp<1：线性过渡

#### 步骤 3：生成位置–频率表

```python
t = torch.arange(end, device=freqs.device)
freqs = torch.outer(t, freqs).float()
freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
```

---

## 四、`apply_rotary_pos_emb` 函数

将预计算的 cos/sin 应用到 Q、K 上。定义位置：`model/model.py` 第 149–163 行。

```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed
```

- `rotate_half`：将 `[a, b]` 变为 `[-b, a]`，实现 2D 旋转变换
- 公式：`x_rotated = x * cos + rotate_half(x) * sin`
- `cos`、`sin` 由 `precompute_freqs_cls` 预计算并按 `position_ids` 索引

---

## 五、在 MokioMindConfig 中的配置

当 `inference_rope_scaling=True` 时，默认启用 YaRN：

```python
rope_scaling = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 16,
    "original_max_position_embeddings": 2048,
    "attention_factor": 1.0,
    "type": "yarn",
}
```

即：以 2048 为训练最大长度，支持约 16× 外推（约 32K），并启用 YaRN 频率插值与注意力缩放。
