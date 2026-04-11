---
title: RMSNorm 归一化
date: 2024-04-11
tags: [大模型, 归一化, transformer]
source: https://github.com/bcefghj/learn-minimind/blob/main/docs/L07-RMSNorm%E5%BD%92%E4%B8%80%E5%8C%96.md
---

# RMSNorm 归一化

## 概述

**RMSNorm**（Root Mean Square Layer Normalization）是 2019 年提出的归一化方法，是 LayerNorm 的简化版本。它去掉了减均值操作，只保留缩放功能。

> 归一化的核心任务是控制数值的**尺度**（magnitude），而不是**位置**（center）

## 与 LayerNorm 的区别

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 减均值 | ✅ 有 | ❌ 去掉 |
| 可学习参数 | γ + β | 仅 γ |
| 计算量 | 算均值+方差 | 只需算均方根 |
| 速度 | 基准 | 快 ~10-15% |

## 计算公式

$$
\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\text{RMS}(x) + \epsilon}
$$

其中：

$$
\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}
$$

等价代码形式：

$$
\text{RMSNorm}(x) = x \cdot \text{rsqrt}\left(\text{mean}(x^2) + \epsilon\right) \cdot \gamma
$$

## 代码实现（MiniMind）

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
```

### 逐行解析

| 代码 | 含义 |
|------|------|
| `x.pow(2)` | 元素平方 |
| `.mean(-1, keepdim=True)` | 沿最后维度求均值 |
| `torch.rsqrt(...)` | 倒数开方 = $1/\sqrt{\cdot}$ |
| `* self.weight` | 乘以可学习参数 γ |

## 优势分析

1. **计算更快**：省去均值计算，速度提升 10-15%
2. **效果几乎持平**：去掉后效果几乎不变
3. **参数量更少**：只有 γ，没有 β
4. **训练稳定**：配合 Pre-Norm 使用，梯度流更顺畅
5. **现代 LLM 主流选择**：GPT-2、LLaMA、Qwen、MiniMind 等均采用

## 使用位置

在 Transformer 中通常有多个 RMSNorm：
- 每个 Transformer 块中 2 个（Attention 前 + FFN 前）
- 以 8 层模型为例：8 × 2 = 16 个
- 再加最终输出 1 个
- **总计 17 个 RMSNorm**

## 参考资料

- [Root Mean Square Layer Normalization (2019)](https://arxiv.org/abs/1910.07467)
- [MiniMind 源码](https://github.com/bcefghj/learn-minimind)
