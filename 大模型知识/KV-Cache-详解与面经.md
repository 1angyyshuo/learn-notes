---
title: KV Cache 详解与面经
date: 2026-05-01
tags: [kv-cache, 推理优化, llm, 面试, paged-attention, mha, gqa, mqa]
category: 大模型工程
status: 完成
---

# KV Cache 详解与面经

## 概述

**KV Cache**（Key-Value Cache）是 Transformer 自回归生成过程中**用显存换时间**的核心技术。它缓存已生成 token 的 Key 和 Value 矩阵，避免每次生成新 token 时重新计算历史序列的注意力。

**一句话理解**：所有已生成 token 的 K/V 都存起来，新 token 的 Q 只跟它们做一次注意力计算。

---

## 1. 为什么需要 KV Cache？

### 1.1 自回归生成的冗余计算

```
生成第 1 个 token：Q[1] 与 K[1]、V[1] 做 attention
生成第 2 个 token：Q[2] 与 K[1-2]、V[1-2] 做 attention  ← K1/V1 重算了一次
生成第 3 个 token：Q[3] 与 K[1-3]、V[1-3] 做 attention  ← K1/V1 重算了两次
...
生成第 N 个 token：Q[N] 与 K[1-N]、V[1-N] 做 attention  ← 前 N-1 对 KV 全部重算
```

**没有 KV Cache**：生成每个 token 时都要跑完整的前向传播重新计算所有历史 KV，计算量变成 `O(N²)` 的平方级增长。

**有 KV Cache**：每个新 token 只需计算自己的 K/V 并存入缓存，注意力计算只对新 token 的 Q 计算结果，计算量降为 `O(N)`。

### 1.2 本质

```
KV Cache 本质：用 GPU 显存空间换取重复计算时间的空间换时间策略

代价：显存占用随序列长度线性增长
收益：避免 O(N²) 的重复计算，生成速度从 O(N²) 降到 O(N)
```

---

## 2. KV Cache 的工作原理

### 2.1 回顾 Attention

```python
# Scaled Dot-Product Attention
Q = X @ W_q  # [batch, seq_len, d_model]
K = X @ W_k  # [batch, seq_len, d_model]
V = X @ W_v  # [batch, seq_len, d_model]

# 注意力计算
scores = Q @ K.T / sqrt(d_k)      # [seq_len, seq_len]
attention_weights = softmax(scores)
output = attention_weights @ V
```

### 2.2 带 KV Cache 的增量推理

```python
# 初始化阶段 (prefill)：一次性编码全部输入 prompt
Q, K, V = proj(prompt_tokens)           # 一次计算所有 prompt 的 Q/K/V
cache_k = K                             # 存入 cache
cache_v = V
output = attention(Q, K, V)             # 输出第一个 token

# 生成阶段 (decode)：每个 token 只算自己的 K/V
for step in range(max_new_tokens):
    Q_new, K_new, V_new = proj(last_token)    # 只算一个 token
    cache_k = concat([cache_k, K_new])        # 追加到 cache
    cache_v = concat([cache_v, V_new])
    output = attention(Q_new, cache_k, cache_v)  # Q=[1], K=[history+1]
    yield output
```

### 2.3 两个阶段

| 阶段 | Prefill（预填充） | Decode（解码） |
|------|-------------------|----------------|
| 输入 | 全部 prompt token | 上一个生成的 token |
| 计算特征 | 矩阵乘法密集，算力受限 | 逐 token 向量运算，带宽受限 |
| KV Cache | 一次性写入 | 每次追加一个 |
| GPU 利用率 | 高 | 低（大量时间花在读取 KV Cache） |

---

## 3. KV Cache 显存计算

### 3.1 精确公式

```
KV Cache 大小 = 2 × B × L × H × D × num_layers × bytes_per_element

其中：
- 2：Key + Value 两份
- B：batch_size（并发序列数）
- L：序列长度（sequence_length）
- H：KV 头数（对于 GQA，H = num_kv_heads）
- D：每个头的维度（head_dim）
- num_layers：Transformer 层数
- bytes_per_element：数据类型字节数（FP16=2, BF16=2, FP32=4）
```

### 3.2 实例计算

以 **Llama-2-7B** 为例（num_layers=32, hidden_size=4096, num_kv_heads=32, head_dim=128, FP16）：

```
B=1, L=2048:
KV Cache = 2 × 1 × 2048 × 32 × 128 × 32 × 2
         = 2 × 1 × 2048 × 32 × 128 × 32 × 2
         = 1,073,741,824 字节 ≈ 1.0 GB

B=1, L=65536 (64K):
KV Cache = 1.0 GB × (65536/2048) ≈ 32 GB
```

以 **Qwen2.5-3B** 为例（num_layers=36, hidden_size=2048, num_kv_heads=2 (GQA), head_dim=128, FP16）：

```
B=1, L=4096:
KV Cache = 2 × 1 × 4096 × 2 × 128 × 36 × 2
         ≈ 151 MB

B=64, L=4096:
KV Cache ≈ 151 MB × 64 ≈ 9.4 GB
```

> 3B 模型用 GQA（kv_heads=2），KV Cache 极小，这也是它能开超大 batch 的原因。

---

## 4. KV Cache 优化技术

### 4.1 MHA → MQA → GQA（架构级优化）

这是减少 KV Cache 最根本的方法——减少 KV 头的数量。

| 方案 | 原理 | KV 头数 | Cache 相对大小 |
|------|------|---------|----------------|
| **MHA**（Multi-Head Attention） | Q/K/V 头数相同 | H | 100% |
| **MQA**（Multi-Query Attention） | 所有 Q 头共享一组 K/V | 1 | 1/H |
| **GQA**（Grouped-Query Attention） | 一组 Q 头共享一组 K/V | g（1 < g < H）| g/H |

```
MHA：  Q1 Q2 Q3 Q4    K1 K2 K3 K4    V1 V2 V3 V4
MQA：  Q1 Q2 Q3 Q4    K1 (共享)      V1 (共享)
GQA：  Q1 Q2 | Q3 Q4  K1 | K2        V1 | V2
       (g=2 组)
```

**面经高频题**：
> Q: Llama 系列为什么从 MHA 变成 MQA 再变成 GQA？
> A: MHA 的 KV Cache 占用太大，MQA 虽最省但生成质量下降。GQA 在两者间取折中，用少数几组 K/V 头尽可能保持质量同时减少显存。Llama-2 70B 用 GQA(g=8) 将 KV Cache 从 MHA 的 100% 降到约 25%。

### 4.2 PagedAttention（vLLM 核心）

vLLM 将操作系统虚拟内存分页机制引入 KV Cache 管理：

```
传统：连续分配
序列 A: |████████████░░░░░░░░░░░░| 预分配 max_len，内部碎片严重

PagedAttention：分页分配
序列 A: Block5 → Block23 → Block7 → Block11
序列 B: Block3 → Block15 → Block9
共享 prefix: Block1 → Block2（AB 共用，Copy-on-Write）
```

**面经高频题**：
> Q: PagedAttention 为什么能提升吞吐量？
> A: (1) 消除内部碎片，KV Cache 利用率从 ~80% 提升到 >95%；(2) 共享前缀的 KV Cache 可以被不同请求复用；(3) 连续批处理调度时不需要预分配和内存移动。

### 4.3 KV Cache 量化（KV-Cache Quantization）

将 KV Cache 从 FP16 压缩到 INT8/INT4/FP8：

| 方案 | 典型效果 | 代表实现 |
|------|----------|----------|
| INT8 量化 | Cache 体积减半，精度损失极小 | vLLM FP8、TensorRT-LLM |
| FP8 量化 | H100 原生支持，无额外 kernel 开销 | H100 的 Transformer Engine |
| INT4 量化 | Cache 体积减到 1/4 | KIVI、FlexGen |
| 非均匀量化 | 保留更多 outlier 精度 | Atom、QServe |

```
量化后的 KV Cache = 原始大小 × (量化位宽 / 原始位宽)

如 INT8 量化：32 × 2048 × 2 GB → 32 × 2048 × 1 GB ≈ 16 GB
```

### 4.4 滑动窗口注意力（Sliding Window Attention）

```python
# 普通 attention：Q_i 关注所有历史的 K_j (j <= i)
# 滑动窗口：Q_i 只关注最近 W 个 token

attn_mask[i, j] = -inf if (i - j) > W else 0

# KV Cache 只需保留最近 W 个 token
```

- Mistral 默认窗口 = 4096
- 长上下文推理时，KV Cache 固定为 W，不随序列长度增长

### 4.5 其他优化策略

| 技术 | 核心思路 | 效果 |
|------|----------|------|
| **Prefix Caching** | 共享 prompt 的 KV Cache 只算一次 | 减少 prefill 计算 |
| **Ring Attention** | 多卡间环形传输 KV 块，分布式处理长序列 | 突破单卡显存限制 |
| **KV Cache Eviction** | 根据注意力分数丢弃不重要的 KV 对 | 动态压缩，保留关键信息 |
| **ALiBi / NoPE** | 相对位置编码替代绝对位置编码 | 消除位置嵌入的额外存储 |
| **Layer-wise Recompute** | 反向传播时按需重算 KV，不常驻显存 | 训练时减少显存占用 |
| **Cross-Layer Sharing** | 相邻层共享 KV Cache | 减少 2x 以上显存（但质量下降） |

---

## 5. 常见面试题（面经）

### Q1：什么是 KV Cache？为什么需要它？

**回答要点**：
自回归生成时，每个新 token 的注意力需要所有历史 token 的 K/V 参与计算。如果没有缓存，每步都要重算全部历史，计算量从 O(N) 暴涨到 O(N²)。KV Cache 将已计算的 K/V 存起来，后续只算新 token 并追加。

### Q2：为什么 KV Cache 只存 K、V，不存 Q？

**回答要点**：

每个 token 的 Q 是"提问者"的角色，只服务于自己的那一步，后续 token 不再需要它：

```
Token 1 生成时：Q1 × (K1.T) 算权重，加权 V1 → 输出 embedding_1
Token 2 生成时：Q2 × (K1, K2).T) 算权重，加权 (V1, V2) → 输出 embedding_2
Token 3 生成时：Q3 × (K1, K2, K3).T) 算权重，加权 (V1, V2, V3) → 输出 embedding_3

观察：token_3 需要 K1、K2、K3 和 V1、V2、V3，但完全不需要 Q1、Q2。
```

**K 和 V 用于"被查询"和"被传递的值"**：
- K 代表这个 token 身上**可供后面 token 查询的特征**
- V 代表这个 token 身上**可供后面 token 提取的价值信息**
- Q 代表当前 token **主动去查询历史**的意图，用一次即弃

**追问：如果某个下游任务需要历史的 Q 怎么办？**
答：绝大多数 decoder-only 自回归生成中，历史 Q 确实不需要。某些 encoder-decoder 场景或双向注意力场景（如 BERT）会保留所有 Q，但那不属于 KV Cache 优化的讨论范围——KV Cache 就是为 decoder-only 增量推理设计的。

### Q3：KV Cache 的大小如何计算？

**回答要点**：
先给出公式 `2 × B × L × num_kv_heads × head_dim × num_layers × dtype_size`，再举例计算，如 7B 模型在 2048 长度下的约 1GB。

**追问：** GQA 的 KV Cache 和 MHA 的 KV Cache 有什么区别？
**答**：GQA 中 K/V 头数更少（如 2 个 vs 32 个），KV Cache 成比例缩小。这是 GQA 设计的核心动机之一。

### Q4：MHA、MQA、GQA 有什么区别？

**回答要点**：
- **MHA**：每个 Q 头对应独立的 K/V 头，KV Cache 最大，质量最高
- **MQA**：所有 Q 头共享一组 K/V，Cache 最小（1/H），质量可能下降
- **GQA**：若干 Q 头分组共享 K/V，在质量和 Cache 大小间折中

Llama-2 70B 用了 8 组 GQA，33% 的 KV Cache 节省，质量基本持平。

### Q5：vLLM 的 PagedAttention 解决了什么问题？原理是什么？

**回答要点**：
传统做法为每个序列预分配最大长度的连续 KV Cache 空间，内部碎片严重且无法共享。PagedAttention 将 KV Cache 切分为固定大小的 Block（默认 16 tokens），通过 Block Table 映射，实现：
1. 按需分配，几乎无内部碎片
2. 非连续存储，利用碎片空间
3. 前缀共享（Copy-on-Write），多请求共享系统 prompt 的 KV Cache
4. 支持连续批处理，随时插入新序列

### Q6：为什么大模型推理是"内存带宽受限"的？

**回答要点**：
Decode 阶段每次只生成一个 token，计算量很小（主要是向量乘法），但需要读取完整模型权重 + KV Cache。读取数据的时间远大于计算时间，GPU 算力大部分在等待数据读取。这就是内存带宽受限。

```
示例：A100
- 算力：312 TFLOPS (FP16)
- 带宽：2 TB/s

生成一个 token 的运算量：~14 TFLOPs（Llama-2-7B）或 ~6 TFLOPs（3B 模型）
从显存读取权重 + KV Cache：远大于计算量
实际吞吐 = 带宽 / (每次生成需读取的数据量)
```

### Q7：Prefill 阶段和 Decode 阶段的瓶颈有什么不同？

**回答要点**：
- **Prefill**：处理完整 prompt，大量矩阵乘法 → **算力受限（Compute-bound）**，GPU 利用率高
- **Decode**：逐 token 生成，注意力变为向量×矩阵 → **带宽受限（Memory-bound）**，大量时间花在读取权重和 KV Cache

优化方向也不同：prefill 要提升算力利用率（如 FlashAttention 减少 HBM 读写），decode 要减少每次生成的数据读取量（如量化、减少 KV 头数）。

### Q8：如何处理超长上下文的 KV Cache 问题？

**回答要点**：
1. **架构层面**：GQA/MQA 减少 KV 头数
2. **量化层面**：KV Cache INT8/INT4 量化，体积缩小 2-4 倍
3. **稀疏层面**：滑动窗口、KV Cache eviction（只保留重要的）
4. **分布式层面**：Ring Attention、Distributed KV Cache
5. **硬件层面**：提高显存容量（H200 141GB）、CPU/NVMe offload

### Q9：KV Cache 量化为什么比权重量化难？

**回答要点**：
权重量化是一次性的——量化后固定。KV Cache 量化是**运行时的**，新 token 不断生成，KV 不断追加：
1. 分布动态变化，无法预估 range
2. 异常值（outlier）在不同 token/层之间分布差异大
3. 需要低开销的在线量化/反量化 kernel
4. 精度损失在长序列中会累积

### Q10：为什么用 FlashAttention 也能省 KV Cache？还是说它其实省的是中间结果？

**回答要点**：
FlashAttention **不节省 KV Cache**。它节省的是 attention 计算过程中的**中间激活值**——它通过分块计算避免了完整 attention matrix（N×N）的存储。但最终的 K 和 V 仍然要存储作为 KV Cache 供后续 token 使用。

```
FlashAttention 节省的：softmax(QK^T) 的中间矩阵（N×N，fp16）
KV Cache 存储的：每层 K 和 V（N×dim），FlashAttention 不影响这个
```

---

## 6. 关键概念速记

| 概念 | 一句话 |
|------|--------|
| KV Cache | 存历史 K/V 避免重算，空间换时间 |
| MHA | Q/K/V 头数相等，Cache 最大 |
| MQA | 所有头共享 K/V，Cache 最小（1/H） |
| GQA | 分组共享 K/V，折中方案 |
| PagedAttention | KV Cache 分页存储，消除碎片 |
| Prefill | 首次编码全部 prompt，算力受限 |
| Decode | 逐 token 生成，内存带宽受限 |
| Prefix Caching | 相同前缀的 KV Cache 只算一次 |
| FP8 KV Cache | H100 原生支持的 Cache 量化方案 |
| Ring Attention | 多卡环形传递 KV 块，分布式处理长序列 |

---

## 关联笔记

- [[vLLM-学习笔记]]
- [[模型量化-AWQ-GPTQ]]
- [[分布式训练基础]]
- [[vLLM-GRPO-LoRA训练适配]]

## 参考资料

- [vLLM Paper: PagedAttention](https://arxiv.org/abs/2309.06180)
- [GQA Paper: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [MQA Paper: Fast Transformer Decoding](https://arxiv.org/abs/1911.02150)
- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)
- [Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889)
