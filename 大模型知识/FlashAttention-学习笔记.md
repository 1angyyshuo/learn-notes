---
title: FlashAttention 学习笔记
date: 2026-04-30
tags: [flashattention, 显存优化, attention, cuda]
category: 大模型工程
status: 进行中
---

# FlashAttention 学习笔记

## 概述

**FlashAttention** 是斯坦福大学提出的高效 Attention 算法，通过**减少 GPU 高带宽内存（HBM）与片上 SRAM 之间的读写次数**，在不牺牲精度的情况下显著加速 Attention 计算并降低显存占用。

- **论文**: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (NeurIPS 2022)
- **作者**: Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, Christopher Ré
- **代码**: https://github.com/Dao-AILab/flash-attention

---

## 1. 核心问题：Attention 的 IO 瓶颈

### 1.1 标准 Attention 的计算流程

```python
# 标准 Self-Attention
Q = X @ W_Q    # (N, d)
K = X @ W_K    # (N, d)
V = X @ W_V    # (N, d)

S = Q @ K.T    # (N, N)  - Score 矩阵
P = softmax(S) # (N, N)  - Attention 权重
O = P @ V      # (N, d)  - 输出
```

### 1.2 GPU 内存层次结构

| 内存层级 | 容量 | 带宽 | 延迟 |
|---------|------|------|------|
| **SRAM（片上缓存）** | ~100KB-200KB | ~19 TB/s | ~1 时钟周期 |
| **HBM（显存）** | 10-80 GB | ~1.5-2 TB/s | ~100 时钟周期 |

**关键洞察**：HBM 带宽远低于计算能力，Attention 的实际瓶颈是**内存读写**，而非计算。

### 1.3 标准 Attention 的显存占用分析

以序列长度 N=4096, head_dim=64, batch=32, heads=12 为例：

| 矩阵 | 形状 | 显存 |
|------|------|------|
| Q, K, V | (N, d) | 3 × 4096 × 64 × 2B = **1.5 MB** |
| S = QK^T | (N, N) | 4096² × 4B = **64 MB** |
| P = softmax(S) | (N, N) | 4096² × 4B = **64 MB** |
| O = PV | (N, d) | 4096 × 64 × 4B = **1 MB** |

总计中间矩阵：**~130 MB**，且随 N² 增长！

**问题**：
1. S 和 P 矩阵需要从 HBM 读写多次
2. 当 N=64K 时，中间矩阵占 **16 GB**，单张 GPU 放不下

---

## 2. FlashAttention 核心思想

### 2.1 IO-Aware 算法设计

传统优化关注**减少 FLOPs**，FlashAttention 关注**减少 HBM 访问次数**。

```
目标：计算 Attention(Q, K, V) 但不将 S 和 P 完整存储到 HBM

思路：
1. 将 Q, K, V 分块（Tiling）
2. 每次将一小块加载到 SRAM
3. 在 SRAM 内完成计算
4. 只将最终结果 O 写回 HBM
```

### 2.2 Softmax 的在线计算

**挑战**：Softmax 需要看到整行才能计算

```
softmax(x_i) = exp(x_i) / sum_j(exp(x_j))

问题：如果分块计算，每个块不知道其他块的值，无法直接 softmax
```

**解决方案：在线 softmax**

分块迭代时维护两个统计量：
- `m`: 当前最大值（用于数值稳定性）
- `l`: 当前 exp 求和

```python
# 块 1 计算
m1 = max(S1)
l1 = sum(exp(S1 - m1))

# 块 2 计算
m2 = max(S2)
l2 = sum(exp(S2 - m2))

# 合并
m_new = max(m1, m2)
l_new = l1 * exp(m1 - m_new) + l2 * exp(m2 - m_new)

# 更新输出
O_new = (O1 * l1 * exp(m1 - m_new) + O2 * l2 * exp(m2 - m_new)) / l_new
```

**效果**：每个块只需加载一次，逐步更新输出，无需存储完整 S 和 P。

### 2.3 Tiling 策略

```
SRAM 大小: M = 100KB (示例)
每个元素: 4 bytes (fp32)

块大小计算:
- Q 块: Br × d
- K 块: Bc × d  
- V 块: Bc × d

约束: 4 × Br × d + 4 × Bc × d + 4 × Br × Bc ≤ M

取 Br = Bc = B:
  4 × B × d + 4 × B × d + 4 × B² ≈ 8Bd + 4B² ≤ M

当 d=64, M=100KB:
  512B + 4B² ≤ 100000
  B ≈ 128
```

**实际分块**（以 A100 为例）：
- SRAM per SM: ~164 KB
- Block size: 64 × 64 或 128 × 128
- 将 (N, N) 的 Attention 分解为 (N/Br, N/Bc) 个小块

---

## 3. 三代演进

### 3.1 FlashAttention-1 (2022)

**核心贡献**：
- 提出 IO-Aware Attention 思想
- Tiling + 在线 Softmax
- 显存从 O(N²) 降至 O(N)

**局限性**：
- 仍有部分非矩阵乘法操作（softmax、mask、dropout）
- 线程块间并行度不够

### 3.2 FlashAttention-2 (2023)

**论文**: [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)

**改进点**：

| 改进 | 说明 | 效果 |
|------|------|------|
| **减少非 matmul FLOPs** | 将 online softmax 的除法移到循环外 | 减少同步 |
| **更好的并行度** | 按 Q 的 row 分配线程块 | 更多并行 |
| **支持 Multi-Query Attention** | 优化 MQA/GQA | 节省显存 |
| **支持 head dim 达 256** | 覆盖 LLaMA-2 等 | 更通用 |

**性能提升**：
- 比 FlashAttention-1 **快 2-4×**
- 接近理论峰值带宽

### 3.3 FlashAttention-3 (2024)

**论文**: [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)

**针对硬件**：NVIDIA Hopper (H100)

**核心创新**：

| 特性 | 说明 |
|------|------|
| **异步执行** | 利用 Hopper 的 Tensor Memory Accelerator (TMA) |
| **FP8 支持** | 原生支持 FP8 低精度计算 |
| **Warp Group Cluster** | 多个 warp group 协作，隐藏延迟 |
| **Softmax 融合** | 将 softmax 的指数计算与矩阵乘法重叠 |

**性能**：
- H100 上达 **1.5-2×** FlashAttention-2
- FP8 下高达 **740 TFLOPS**

---

## 4. 显存与复杂度分析

### 4.1 显存对比

| 方法 | 中间矩阵显存 | 总显存 | 精度 |
|------|------------|--------|------|
| 标准 Attention | O(N²) | 高 | 精确 |
| **FlashAttention** | **O(N)** | **低** | **精确** |
| 稀疏 Attention | O(N) | 低 | 近似 |

**注意**：FlashAttention **不减少计算量**（仍是 O(N²d)），但大幅减少显存和 IO。

### 4.2 显存估算公式

```python
# 标准 Attention 显存
memory_naive = 4 * N * N * sizeof(dtype)  # S + P 矩阵

# FlashAttention 显存
memory_flash = 2 * N * d * sizeof(dtype)  # 只需存 O 输出

# 节省比例
saving_ratio = (N * N) / (N * d / 2) = 2N / d

# N=4096, d=64: 节省 ~128 倍
# N=32768, d=128: 节省 ~512 倍
```

### 4.3 实际使用中的显存分配

```
总显存 = 模型权重 + KV Cache + Activation + 中间矩阵

FlashAttention 影响的是 "中间矩阵" 部分：
- 标准：中间矩阵 ~ N²
- FlashAttention：中间矩阵 ~ N (可忽略)
```

---

## 5. 实践使用

### 5.1 安装

```bash
# PyPI 安装 (预编译 wheel)
pip install flash-attn --no-build-isolation

# 从源码编译 (需要 CUDA)
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install

# FlashAttention-3 (Hopper 专用)
pip install flash-attn>=2.6.0
```

**环境要求**：
- CUDA 11.6+ / 12.x
- PyTorch 1.12+
- NVIDIA GPU (Ampere, Ada, Hopper)

### 5.2 PyTorch 中使用

```python
from flash_attn import flash_attn_func

# 输入: (batch, seqlen, nheads, headdim)
q = torch.randn(2, 4096, 12, 128, device='cuda', dtype=torch.float16)
k = torch.randn(2, 4096, 12, 128, device='cuda', dtype=torch.float16)
v = torch.randn(2, 4096, 12, 128, device='cuda', dtype=torch.float16)

# 使用 FlashAttention
out = flash_attn_func(q, k, v, causal=True)  # causal mask 用于自回归

# out shape: (2, 4096, 12, 128)
```

### 5.3 与 HuggingFace 集成

```python
from transformers import AutoModelForCausalLM

# 大多数现代模型已自动使用 FlashAttention
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
    device_map="auto",
)

# 或者通过 config
from transformers import AutoConfig

config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
config._attn_implementation = "flash_attention_2"
```

### 5.4 实际性能对比

```python
import torch
import time

# 测试配置
batch = 8
heads = 12
d = 64
seq_lens = [1024, 2048, 4096, 8192, 16384]

def benchmark(fn, q, k, v, warmup=10, repeat=50):
    # Warmup
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(repeat):
        fn(q, k, v)
    torch.cuda.synchronize()
    return (time.time() - start) / repeat * 1000  # ms

for N in seq_lens:
    q = torch.randn(batch, heads, N, d, device='cuda', dtype=torch.float16)
    k = torch.randn(batch, heads, N, d, device='cuda', dtype=torch.float16)
    v = torch.randn(batch, heads, N, d, device='cuda', dtype=torch.float16)
    
    # 标准 Attention
    time_naive = benchmark(
        lambda q,k,v: torch.softmax(q @ k.transpose(-2,-1) / (d**0.5), dim=-1) @ v,
        q, k, v
    )
    
    # FlashAttention
    from flash_attn import flash_attn_func
    q_f = q.transpose(1, 2).contiguous()  # FA 需要 (b, s, h, d)
    k_f = k.transpose(1, 2).contiguous()
    v_f = v.transpose(1, 2).contiguous()
    time_fa = benchmark(flash_attn_func, q_f, k_f, v_f)
    
    print(f"N={N:5d}: Naive={time_naive:6.2f}ms, FlashAttn={time_fa:6.2f}ms, Speedup={time_naive/time_fa:.2f}x")

# 典型输出 (A100):
# N= 1024: Naive=  5.2ms, FlashAttn=  1.8ms, Speedup=2.89x
# N= 2048: Naive= 18.5ms, FlashAttn=  5.2ms, Speedup=3.56x
# N= 4096: Naive= 72.3ms, FlashAttn= 18.1ms, Speedup=3.99x
# N= 8192: Naive=280.1ms, FlashAttn= 65.4ms, Speedup=4.28x
# N=16384: OOM,           FlashAttn=240.2ms, Speedup=inf
```

---

## 6. 常见问题与限制

### 6.1 不支持的情况

| 限制 | 说明 | 解决方案 |
|------|------|----------|
| head_dim > 256 | 当前不支持 | 用标准 Attention |
| ALiBi 位置编码 | 部分版本不支持 | 检查版本或使用其他编码 |
| attention_mask | 只支持因果/全可见 | 自定义 mask 需回退 |
| CPU | 仅支持 CUDA | 用标准 Attention |

### 6.2 什么时候不需要 FlashAttention？

- **短序列** (N < 512)：加速不明显，安装成本可能不划算
- **非自回归**：如 BERT 编码器，batch 小加速有限
- **非 NVIDIA GPU**：仅支持 CUDA

---

## 7. 高频面试题

### 概念理解题

**Q1: FlashAttention 解决了什么问题？**
> **答**：标准 Attention 的瓶颈不是计算，而是 **HBM 和 SRAM 之间的 IO**。FlashAttention 通过 Tiling 和在线 Softmax，避免将 O(N²) 的中间矩阵写回 HBM，将显存从 O(N²) 降到 O(N)，同时不损失精度。

**Q2: FlashAttention 减少计算量吗？**
> **答**：**不减少**。FLOPs 仍然是 O(N²d)。它减少的是 **HBM 读写次数** 和 **显存占用**，属于 IO-Aware 优化而非计算优化。

**Q3: 为什么 FlashAttention 是"精确"的，而稀疏 Attention 是"近似"的？**
> **答**：FlashAttention 计算的是**完全相同的 Attention 输出**，只是改变了计算的顺序和内存访问模式。稀疏 Attention 则丢弃了部分 token 对，计算的是近似值。

### 技术细节题

**Q4: 在线 Softmax 怎么实现的？**
> **答**：分块迭代时维护当前的最大值 `m` 和指数和 `l`。当处理新块时，用新的最大值更新旧的统计量，通过 `exp(m_old - m_new)` 调整比例。最终输出逐步累加，不需要存储完整 softmax 矩阵。

**Q5: Tiling 的块大小怎么选？**
> **答**：受限于 GPU SRAM 大小。假设 SRAM 为 M，块大小 B 需满足 `8Bd + 4B² ≤ M`。实际实现中会针对具体 GPU 架构调优（如 A100 用 128×128）。

**Q6: FlashAttention-2 相比 v1 的核心改进是什么？**
> **答**：
> 1. 减少非 matmul FLOPs（将 softmax 归一化移到循环外）
> 2. 更好的 work partitioning（按 row 并行）
> 3. 支持 MQA/GQA
> 4. 支持更大的 head_dim (256)

### 对比分析题

**Q7: FlashAttention vs vLLM PagedAttention 的区别？**
> **答**：
> - **FlashAttention**：优化单个 Attention 算子的**内存访问模式**，减少中间矩阵的 HBM 读写
> - **PagedAttention**：优化**服务层**的 KV Cache 管理，通过分页减少显存碎片和实现连续批处理
> - 两者可以**叠加使用**：vLLM 内部使用 FlashAttention 加速 Attention 计算

**Q8: FlashAttention vs 稀疏 Attention（如 Quest）的区别？**
> **答**：
> - FlashAttention：不稀疏任何计算，只是**重排计算顺序**以优化 IO，结果精确
> - Quest/稀疏 Attention：**跳过部分 token 对的计算**，结果近似，但可能进一步减少计算量

### 场景计算题

**Q9: 计算 N=8192, d=128, batch=4, heads=16 时，标准 Attention 和 FlashAttention 的显存差异。**
> **答**：
> - 标准 Attention：S + P = 2 × 4 × 16 × 8192² × 2B = **13.4 GB** (fp16)
> - FlashAttention：仅需 O 输出 + 少量 buffer ≈ **0.5 GB**
> - 节省约 **26 倍**

**Q10: 如果 SRAM 是 100KB，d=64，求最大块大小 B。**
> **答**：约束 `8Bd + 4B² ≤ 100000`，代入 d=64 得 `512B + 4B² ≤ 100000`，解得 B ≈ **125**，实际取 **64 或 128**（对齐）。

### 延伸思考

**Q11: 为什么 FlashAttention-3 需要 Hopper 架构？**
> **答**：FA-3 利用 Hopper 特有的 **Tensor Memory Accelerator (TMA)** 实现异步数据加载，以及 **Warp Group Cluster** 实现跨 warp 协作。这些特性在 Ampere/Ada 上不存在。

**Q12: 如何在已有模型中接入 FlashAttention？**
> **答**：
> 1. 最简单：`transformers` 库中设置 `attn_implementation="flash_attention_2"`
> 2. 手动修改：将 `nn.MultiheadAttention` 或自定义 Attention 替换为 `flash_attn_func`
> 3. 需注意输入 shape 要求：(batch, seqlen, nheads, headdim)

---

## 参考资料

- [FlashAttention 论文 (NeurIPS 2022)](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3 论文](https://arxiv.org/abs/2407.08608)
- [官方 GitHub 仓库](https://github.com/Dao-AILab/flash-attention)
- [Tri Dao 博客: FlashAttention 详解](https://princeton-nlp.github.io/flash-attention/)
- [Lilian Weng: Attention Mechanisms](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)

---

*下一步：建议阅读 FlashAttention 源码中的 `flash_attn_func` 实现，理解 Tiling 和 online softmax 的 CUDA Kernel 代码。*
