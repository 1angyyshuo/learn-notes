---
title: RoPE 位置编码与外推
date: 2026-04-30
tags: [大模型, 位置编码, RoPE, 注意力机制, 长度外推]
---

# RoPE 位置编码与外推

## 概述

**RoPE (Rotary Position Embedding，旋转位置编码)** 是目前大语言模型（如 LLaMA、Qwen、Baichuan 等）最主流的位置编码方案。它通过**旋转矩阵**将位置信息注入到注意力计算中，兼具绝对位置编码的简洁性和相对位置编码的优良外推特性。

**核心问题**：Transformer 的自注意力机制本身对 token 顺序不敏感（置换不变性），必须引入位置编码来让模型感知序列顺序。

---

## 详细内容

### 1. 位置编码的演进

#### 1.1 为什么需要位置编码？

自注意力计算：

$$
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

如果不加位置信息，交换两个 token 的位置不会改变注意力输出。模型必须知道 "第 5 个词" 和 "第 10 个词" 的区别。

#### 1.2 绝对位置编码 (Absolute PE)

**原始 Transformer (Sinusoidal)**：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

- 直接加到词嵌入上：`x = embedding + PE(pos)`
- **问题**：只能内插（train短test短），外推（train短test长）效果差

**可学习绝对位置编码**：BERT 的做法，直接学习一个位置嵌入矩阵。
- **问题**：长度固定，无法处理更长序列

#### 1.3 相对位置编码 (Relative PE)

核心思想：模型不需要知道绝对位置，只需要知道 token 之间的**相对距离**。

**T5 的 Relative Bias**：在注意力分数上加上一个与相对距离相关的偏置项。

**ALiBi (Attention with Linear Biases)**：
- 不给 embedding 加位置信息
- 直接在注意力分数上减去一个与距离成正比的惩罚项：$q_i k_j^T - m \cdot |i - j|$
- **优点**：天然具有良好的外推性

---

### 2. RoPE 旋转位置编码

#### 2.1 核心思想

RoPE 将位置编码融入到**查询 (Query) 和键 (Key) 向量**中，通过**旋转矩阵**实现：

> 将二维向量 $(x, y)$ 旋转角度 $\theta \cdot pos$，这样两个向量的内积只依赖于它们的**相对位置差**。

#### 2.2 数学推导

对于第 $m$ 个位置的二维词嵌入 $(x_m^{(1)}, x_m^{(2)})$，应用旋转：

$$
\begin{pmatrix} q_m^{(1)} \\ q_m^{(2)} \end{pmatrix} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_m^{(1)} \\ x_m^{(2)} \end{pmatrix}
$$

扩展到 $d$ 维（每两个维度一组，共 $d/2$ 组）：

$$
f(q, m) = R_{\Theta, m} \cdot q
$$

其中 $R_{\Theta, m}$ 是块对角旋转矩阵：

$$
R_{\Theta, m} = \begin{pmatrix}
\cos(m\theta_1) & -\sin(m\theta_1) & 0 & 0 & \cdots \\
\sin(m\theta_1) & \cos(m\theta_1) & 0 & 0 & \cdots \\
0 & 0 & \cos(m\theta_2) & -\sin(m\theta_2) & \cdots \\
0 & 0 & \sin(m\theta_2) & \cos(m\theta_2) & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix}
$$

#### 2.3 频率设置 (RoPE Theta)

每组维度使用不同的旋转频率：

$$
\theta_i = 10000^{-2(i-1)/d}, \quad i \in [1, 2, \ldots, d/2]
$$

- **低频**（小的 $i$）：大的 $\theta$，对应长波长，感知长距离依赖
- **高频**（大的 $i$）：小的 $\theta$，对应短波长，感知细粒度位置差异

**LLaMA 3 将 base 从 10000 改为 500000**，就是为了更好地支持长上下文。

#### 2.4 关键性质：相对位置等价性

RoPE 的核心优势在于：

$$
\langle f(q, m), f(k, n) \rangle = g(q, k, m - n)
$$

即两个旋转后的向量的内积**只依赖于它们的相对距离** $(m - n)$，而不是绝对位置！

这使得 RoPE 同时具有：
- **绝对位置编码的简洁性**：每个 token 独立计算，易于实现
- **相对位置编码的优点**：注意力分数天然体现相对距离

---

### 3. 长度外推问题 (Length Extrapolation)

#### 3.1 什么是外推？

- **内插 (Interpolation)**：在训练时见过的长度范围内处理（如训练 2048，测试 1024）
- **外推 (Extrapolation)**：处理比训练时**更长**的序列（如训练 2048，测试 8192）

**外推的核心挑战**：

1. **未见过的位置编码值**：绝对位置编码在训练时只见过 $[0, 2048]$ 的位置，测试时出现 $[2048, 8192]$ 的位置编码是模型从未见过的
2. **注意力分散**：RoPE 中高频分量在训练长度之外周期性混乱，导致注意力分数失真

#### 3.2 为什么 RoPE 外推困难？

对于位置 $m$ 和 $n$，注意力分数依赖于 $\cos((m-n)\theta_i)$ 和 $\sin((m-n)\theta_i)$。

当 $|m - n|$ 超过训练时的最大距离 $L_{max}$：
- 低频分量：波长很长，仍然能正常变化
- **高频分量**：波长很短，$|m - n|\theta_i$ 可能远超训练范围，导致 $\cos/\sin$ 值混乱

这导致远距离 token 的注意力分数变得不可预测。

---

### 4. 外推方法演进

#### 4.1 Position Interpolation (PI)

**核心思想**：将位置编码**压缩**到训练时见过的范围内。

原本位置 $m$ 现在映射到 $m \cdot \frac{L_{train}}{L_{target}}$：

```
如果训练长度 2048，目标长度 8192：
位置 8192 的位置编码 → 使用位置 2048 的编码值
位置 4096 的位置编码 → 使用位置 1024 的编码值
```

**优点**：简单有效，只需要少量微调
**缺点**：
- 所有位置都被压缩，损失了局部区分度
- 高频信息被过度压缩

#### 4.2 NTK-Aware Scaling (NTK 感知扩展)

**核心思想**：不改位置索引，而是**调整 RoPE 的 base (频率)**，让所有波长在更长序列上都能正常工作。

**NTK (Neural Tangent Kernel) 理论**指出：修改 base 可以改变模型感知的空间维度。

修改后的 base：

$$
\theta'_i = \left(\lambda \cdot 10000\right)^{-2i/d}
$$

其中 $\lambda = \left(\frac{L_{target}}{L_{train}}\right)^{d/(d-2)}$

**优点**：无需微调即可实现一定外推
**缺点**：高频分量仍然可能有问题

#### 4.3 Dynamic NTK

**核心思想**：根据**当前序列长度动态调整** base。

```python
# 伪代码
if current_length <= train_length:
    use_base = 10000
else:
    scale = current_length / train_length
    use_base = 10000 * scale^(d/(d-2))
```

**优点**：自适应不同长度，实际效果好
**缺点**：实现稍复杂

#### 4.4 YaRN (Yet another RoPE extensioN)

目前最主流的外推方案之一，结合了 PI 和 NTK 的优点。

**核心思想**：
1. **频率分组处理**：对不同频率分量采用不同策略
2. **温度缩放 (Temperature Scaling)**：对注意力分数乘以温度系数 $t$，防止长序列 softmax 过于平坦
3. **NTK 扩展**：扩展 base

公式：

$$
Attention = softmax\left(\frac{q_m k_n^T}{t \cdot \sqrt{d_k}}\right)
$$

其中 $t$ 是根据扩展比例计算的注意力温度。

**优点**：
- 支持 2x, 4x, 8x 甚至 16x 扩展
- 只需要少量微调（如 0.1% 的训练数据）

#### 4.5 方法对比

| 方法 | 是否需要微调 | 扩展倍数 | 实现复杂度 | 代表模型 |
|------|-------------|---------|-----------|---------|
| 直接外推 | 否 | < 1.5x | 简单 | 早期模型 |
| Position Interpolation | 是 (少量) | 2-8x | 简单 | - |
| NTK-Aware | 否 | 2-4x | 中等 | KoboldAI |
| Dynamic NTK | 否 | 2-4x | 中等 | - |
| YaRN | 是 (少量) | 8-16x | 中等 | LLaMA 2 Long |
| 修改 RoPE Base | 预训练阶段 | 16x+ | 简单 | LLaMA 3 (base=500000) |

---

### 5. 代码实现要点

#### 5.1 RoPE 的高效实现

现代实现（如 Transformers 库）使用**复数乘法**来高效计算：

```python
# 核心思想：将 (x, y) 视为复数 x + yi，旋转 = 乘以 e^(i*m*theta)
# 现代 GPU 可以用旋转因子直接做向量乘法

# 预计算 cos, sin
cos = torch.cos(position * theta)  # [seq_len, dim/2]
sin = torch.sin(position * theta)  # [seq_len, dim/2]

# 应用旋转 (使用 -x[..., 1::2], x[..., ::2] 的技巧)
x1 = x[..., ::2]   # 偶数维
x2 = x[..., 1::2]  # 奇数维

rotated_x = torch.stack([
    x1 * cos - x2 * sin,
    x1 * sin + x2 * cos
], dim=-1).flatten(-2)
```

#### 5.2 FlashAttention + RoPE

现代大模型通常在**注意力计算之前**应用 RoPE（即 pre-RoPE）：

```python
# 标准流程
q = apply_rope(q, position_ids)  # [batch, seq_len, dim]
k = apply_rope(k, position_ids)
out = flash_attention(q, k, v)   # 内部不需要位置信息
```

FlashAttention 内部只计算 $qk^T$，位置信息已经通过 RoPE 编码进 $q, k$ 中。

---

## 关联笔记

- [[FlashAttention-学习笔记]] — 现代注意力计算优化，与 RoPE 配合使用
- [[稀疏注意力-学习笔记]] — 长序列场景下的注意力优化
- [[vLLM-学习笔记]] — 推理引擎中的位置编码处理

## 参考资料

1. **RoPE 原始论文**: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
2. **Position Interpolation**: [Extending Context Window of Large Language Models via Position Interpolation](https://arxiv.org/abs/2306.15595)
3. **NTK-Aware Scaling**: [NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)
4. **YaRN**: [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)
5. **ALiBi**: [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)
6. **代码参考**: [HuggingFace Transformers - RoPE 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
