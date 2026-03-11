# RMSNorm 模块分析

## 概述

**RMSNorm**（Root Mean Square Layer Normalization）是一种简化版 Layer Normalization，仅基于均方根统计进行正则化，相比 LayerNorm 去掉了均值中心化，计算更轻量、速度更快，在 LLaMA、Mistral 等主流大模型中广泛使用。

定义位置：`model/model.py` 第 80–91 行。

**参考文献**：Biao Zhang & Rico Sennrich, *Root Mean Square Layer Normalization*, NeurIPS 2019.

---

## 与 LayerNorm 的对比

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 中心化 | 先减均值 | 不做中心化 |
| 缩放 | 用标准差 | 用均方根 (RMS) |
| 可学习参数 | γ（缩放）、β（偏移） | 仅 γ（缩放） |
| 计算量 | 更大 | 更小（约 7%–64% 加速） |

RMSNorm 假设 LayerNorm 的「中心不变性」不是必需的，只保留「缩放不变性」即可获得相当性能。

---

## 函数实现

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
```

---

## 数学形式

设输入 \( x \in \mathbb{R}^{d} \)，RMSNorm 定义为：

\[
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\text{RMS}(x) + \epsilon}
\]

其中均方根：

\[
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2} = \sqrt{\text{mean}(x^2)}
\]

代码中通过 \( \text{rsqrt}(z) = 1/\sqrt{z} \) 实现：

\[
\frac{x}{\text{RMS}(x) + \epsilon} = x \cdot \frac{1}{\sqrt{\text{mean}(x^2) + \epsilon}}
\]

---

## 参数说明

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `dim` | int | - | 特征维度，与 `weight` 一致 |
| `eps` | float | 1e-5 | 数值稳定项，避免分母为 0 |

**可学习参数**：`weight`（γ），形状 `[dim]`，初始为全 1。

---

## 计算流程

1. **`_norm(x)`**
   - `x.pow(2)`：逐元素平方  
   - `.mean(-1, keepdim=True)`：沿最后一维求均值  
   - 加 `eps` 后 `torch.rsqrt` 得到 \( 1/\sqrt{\text{mean}(x^2)+\epsilon} \)  
   - 与 `x` 相乘，得到归一化后的向量（均值不一定为 0）

2. **`forward(x)`**
   - 先将 `x` 转为 `float()` 做数值稳定计算  
   - 调用 `_norm` 得到归一化结果  
   - 与可学习权重 `self.weight` 逐元素相乘  
   - 用 `.type_as(x)` 恢复原始数据类型（如 bf16/half）

---

## 在 Transformer 中的典型用法

RMSNorm 常替代 LayerNorm，用于：

- 每个 Transformer 块前/后的归一化（Pre-LN / Post-LN）
-  Attention 子层后的残差分支
- FFN 子层后的残差分支

本实现与 LLaMA、Mistral 等模型中的 RMSNorm 用法一致。
