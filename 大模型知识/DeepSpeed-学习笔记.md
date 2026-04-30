---
title: DeepSpeed 学习笔记
date: 2026-04-30
tags: [deepspeed, 分布式训练, ZeRO, 大模型训练]
category: 大模型工程
status: 进行中
---

# DeepSpeed 学习笔记

## 概述

**DeepSpeed** 是微软开源的深度学习训练优化库，核心贡献是 **ZeRO（Zero Redundancy Optimizer）**，通过在数据并行进程中分片存储 Optimizer States、Gradients 和 Parameters，大幅降低大模型训练所需的显存。

- **GitHub**: https://github.com/microsoft/DeepSpeed
- **文档**: https://www.deepspeed.ai/
- **核心论文**: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (SC 2020)

---

## 1. 核心问题：大模型训练的显存瓶颈

### 1.1 训练时的显存占用分析

以 Adam 优化器、混合精度训练为例，显存占用包括：

| 组成部分 | 公式 | 7B 模型 (fp16) | 70B 模型 (fp16) |
|---------|------|----------------|-----------------|
| **模型参数 (Params)** | 2 × P | 14 GB | 140 GB |
| **梯度 (Gradients)** | 2 × P | 14 GB | 140 GB |
| **Optimizer States** | 12 × P (Adam: 2×fp32 copy + 2×momentum) | 84 GB | 840 GB |
| **激活值 (Activations)** | 取决于 batch size 和序列长度 | ~varies | ~varies |
| **总计** | **~16P + Activations** | **~112 GB+** | **~1120 GB+** |

**问题**：
- 7B 模型单卡就需要 112GB+，A100 80GB 都放不下
- 70B 模型需要 1TB+ 显存，完全不现实
- 主要瓶颈不是参数本身，而是 **Optimizer States**

### 1.2 并行策略回顾

| 并行方式 | 切分对象 | 解决的问题 | 通信量 |
|---------|---------|-----------|--------|
| **数据并行 (DP)** | 数据 batch | 加速训练 | 每次迭代同步梯度 |
| **模型并行 (MP/TP)** | 模型层/参数 | 模型太大放不进单卡 | 每层前向/反向通信 |
| **流水线并行 (PP)** | 模型层组 | 层数太多 | 每阶段传递激活值 |
| **ZeRO-DP** | Optimizer/Gradient/Param | 显存冗余 | 按需通信参数 |

**关键洞察**：数据并行时，每张 GPU 都保存完整的 Optimizer States、Gradients 和 Parameters，这是**冗余**的！

---

## 2. ZeRO：零冗余优化器

### 2.1 核心思想

```
数据并行（传统）：
GPU 0: [Param, Gradient, Optimizer]  ← 完整副本
GPU 1: [Param, Gradient, Optimizer]  ← 完整副本
GPU 2: [Param, Gradient, Optimizer]  ← 完整副本
GPU 3: [Param, Gradient, Optimizer]  ← 完整副本
→ 4× 冗余

ZeRO（分片）：
GPU 0: [Param[0], Gradient[0], Optimizer[0]]  ← 1/4
GPU 1: [Param[1], Gradient[1], Optimizer[1]]  ← 1/4
GPU 2: [Param[2], Gradient[2], Optimizer[2]]  ← 1/4
GPU 3: [Param[3], Gradient[3], Optimizer[3]]  ← 1/4
→ 0 冗余，需要时通信
```

### 2.2 ZeRO 三阶段

#### ZeRO-1：Optimizer States 分片

```python
# 显存占用: 2P (params) + 2P (gradients) + 12P/N (optimizer)
# N = GPU 数量

每张 GPU 保存：
- 完整参数 (fp16): 2P
- 完整梯度 (fp16): 2P
- 1/N Optimizer States (fp32): 12P/N

显存节省：16P → 4P + 12P/N
N=8: 16P → 4P + 1.5P = 5.5P  (节省 3×)
```

**适用场景**：Optimizer States 是主要瓶颈时

#### ZeRO-2：Optimizer States + Gradients 分片

```python
# 显存占用: 2P (params) + 2P/N (gradients) + 12P/N (optimizer)

每张 GPU 保存：
- 完整参数 (fp16): 2P
- 1/N 梯度 (fp16): 2P/N
- 1/N Optimizer States (fp32): 12P/N

显存节省：16P → 2P + 14P/N
N=8: 16P → 2P + 1.75P = 3.75P  (节省 4.3×)
```

**适用场景**：梯度也占用大量显存时

#### ZeRO-3：Optimizer States + Gradients + Parameters 分片

```python
# 显存占用: 2P/N (params) + 2P/N (gradients) + 12P/N (optimizer)

每张 GPU 保存：
- 1/N 参数 (fp16): 2P/N
- 1/N 梯度 (fp16): 2P/N
- 1/N Optimizer States (fp32): 12P/N

显存节省：16P → 16P/N
N=8: 16P → 2P  (节省 8×)
N=64: 16P → 0.25P
```

**适用场景**：模型参数本身太大，单卡放不下

### 2.3 显存节省对比

| ZeRO 阶段 | 单卡显存 | 节省倍数 | 7B 模型 (N=8) | 70B 模型 (N=64) |
|-----------|---------|---------|---------------|-----------------|
| 无 ZeRO | 16P | 1× | ~112 GB | ~1120 GB |
| **ZeRO-1** | 4P + 12P/N | ~3× | **~38 GB** | ~160 GB |
| **ZeRO-2** | 2P + 14P/N | ~4× | **~27 GB** | ~115 GB |
| **ZeRO-3** | 16P/N | N× | **~14 GB** | **~18 GB** |

### 2.4 通信开销分析

```
ZeRO-1: 
  - 前向：无额外通信
  - 反向：AllReduce 梯度 (与标准 DP 相同)
  - 优化：AllGather optimizer states
  → 通信量 ~1.5× 标准 DP

ZeRO-2:
  - 反向：Reduce-Scatter 梯度 (替代 AllReduce)
  - 优化：AllGather optimizer states
  → 通信量 ~1.5× 标准 DP

ZeRO-3:
  - 前向：AllGather 参数
  - 反向：AllGather 参数 + Reduce-Scatter 梯度
  - 优化：AllGather optimizer states
  → 通信量 ~3× 标准 DP
```

**关键权衡**：显存节省 vs 通信开销。ZeRO-3 显存节省最多，但通信量也最大。

---

## 3. ZeRO-Infinity 与 Offload

### 3.1 NVMe Offload

当 GPU 显存不够时，将 Optimizer States 和参数 offloading 到 **CPU 内存** 甚至 **NVMe SSD**。

```python
# DeepSpeed Config
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",      # 或 "nvme"
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",      # 或 "nvme"
      "pin_memory": true
    }
  }
}
```

**显存节省**：
- GPU 上只需保留当前计算所需的参数块
- 70B 模型可以在 **单张 V100 (32GB)** 上训练！

**代价**：
- CPU↔GPU 或 NVMe↔CPU 的数据传输延迟
- 训练速度显著下降（可能 2-5× 慢）
- 适合显存极度受限的场景

### 3.2 ZeRO-Infinity

**论文**: [ZeRO-Infinity: Breaking GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857) (SC 2021)

**核心创新**：
1. **Infinity offload engine**: 自动管理 GPU/CPU/NVMe 间的数据流动
2. **3D parallelism**: DP + MP + PP 的自动组合
3. **Bandwidth-centric partitioning**: 根据带宽优化分片策略

**效果**：
- 可以训练 **万亿参数** 模型
- 支持 **512 张 GPU** 以上的规模

---

## 4. 其他 DeepSpeed 特性

### 4.1 DeepSpeed-MoE

专家混合模型 (Mixture of Experts) 的训练和推理优化：

```python
# 将 Transformer FFN 层替换为 MoE 层
from deepspeed.moe.layer import MoE

moe_layer = MoE(
    hidden_size=hidden_dim,
    expert=expert_module,
    num_experts=64,        # 专家数量
    ep_size=8,             # 专家并行度
    k=2,                   # Top-K 路由
)
```

**优势**：
- 模型容量增加，但计算量不变（每次只激活部分专家）
- DeepSpeed 自动优化 All-to-All 通信

### 4.2 DeepSpeed-Inference

推理阶段优化：

```python
import deepspeed

# 加载模型
model = ...

# 使用 DeepSpeed Inference
model = deepspeed.init_inference(
    model,
    mp_size=2,                    # 模型并行度
    dtype=torch.float16,
    replace_with_kernel_inject=True,  # 注入优化 kernel
    enable_cuda_graph=True,       # CUDA Graph 加速
)
```

**优化**：
- Kernel 融合（fused attention, fused MLP）
- INT8/FP16 量化
- CUDA Graph 减少 CPU 开销

### 4.3 DeepSpeed-Compression

压缩训练：
- **1-bit Adam**: 梯度压缩到 1-bit，减少通信
- **Curriculum Learning**: 逐步增加序列长度
- **Progressive Layer Dropping**: 训练时随机丢弃层

---

## 5. 实践使用

### 5.1 安装

```bash
# PyPI 安装
pip install deepspeed

# 从源码安装 (推荐，获取最新特性)
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
DS_BUILD_OPS=1 pip install -e .

# 验证安装
ds_report
```

### 5.2 配置 ds_config.json

```json
{
  "train_batch_size": 64,
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 1,
  
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 5e-5,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 5e-5,
      "warmup_num_steps": 500
    }
  },
  
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  
  "gradient_clipping": 1.0,
  
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  
  "wall_clock_breakdown": false
}
```

### 5.3 训练脚本

```python
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. 准备数据
train_dataset = ...
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=8,
    shuffle=True
)

# 3. 初始化 DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"
)

# 4. 训练循环
for epoch in range(3):
    for batch in train_dataloader:
        # 数据放到当前设备
        input_ids = batch["input_ids"].to(model_engine.local_rank)
        labels = batch["labels"].to(model_engine.local_rank)
        
        # 前向传播
        outputs = model_engine(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # 反向传播
        model_engine.backward(loss)
        
        # 梯度更新
        model_engine.step()
        
        # 打印日志 (只在 rank 0)
        if model_engine.global_rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 5. 保存模型 (只在 rank 0)
if model_engine.global_rank == 0:
    model_engine.save_checkpoint("./checkpoints")
```

### 5.4 多节点训练启动

```bash
# 使用 deepspeed 启动器
# 节点 0 (主节点)
deepspeed --num_gpus=8 --num_nodes=2 --master_addr="192.168.1.1" \
    --master_port=29500 \
    train.py --deepspeed ds_config.json

# 节点 1
deepspeed --num_gpus=8 --num_nodes=2 --master_addr="192.168.1.1" \
    --master_port=29500 --node_rank=1 \
    train.py --deepspeed ds_config.json

# 或者使用 torchrun + hostfile
deepspeed --hostfile=hostfile train.py --deepspeed ds_config.json

# hostfile 内容:
# worker-0 slots=8
# worker-1 slots=8
```

### 5.5 显存监控

```python
# DeepSpeed 自动记录显存使用
# 在 config 中开启:
{
  "wandb": {
    "enabled": true,
    "project": "my-project"
  },
  "tensorboard": {
    "enabled": true,
    "output_path": "./tb_logs"
  }
}

# 或在代码中手动查看
from deepspeed.runtime.utils import see_memory_usage
see_memory_usage("After forward", force=True)
```

---

## 6. 与其他框架对比

### 6.1 DeepSpeed vs Megatron-LM vs FSDP

| 特性 | DeepSpeed | Megatron-LM | PyTorch FSDP |
|------|-----------|-------------|--------------|
| **开发者** | 微软 | NVIDIA | Meta/PyTorch |
| **核心优化** | ZeRO (显存分片) | Tensor Parallel (TP) | Sharded DDP |
| **易用性** | 高 (config 驱动) | 中 (需改模型) | 高 (PyTorch 原生) |
| **最大模型** | 万亿级 (ZeRO-Infinity) | 数万亿 (3D并行) | 百亿级 |
| **MoE 支持** | 原生支持 | 支持 | 需手动实现 |
| **推理优化** | DeepSpeed-Inference | FasterTransformer | torch.compile |
| **推荐场景** | 研究/通用 | 大规模预训练 | PyTorch 生态 |

### 6.2 如何选择？

```
场景 1: 单节点 8 卡，7B-13B 模型微调
→ FSDP 或 DeepSpeed ZeRO-2

场景 2: 多节点，70B+ 模型预训练
→ DeepSpeed ZeRO-3 + Megatron TP/PP

场景 3: 显存极度受限，单卡训练 7B+
→ DeepSpeed ZeRO-3 + Offload

场景 4: MoE 模型训练
→ DeepSpeed-MoE

场景 5: 生产环境推理优化
→ DeepSpeed-Inference 或 vLLM
```

---

## 7. 高频面试题

### 概念理解题

**Q1: 为什么数据并行时会有显存冗余？**
> **答**：
> 数据并行每张 GPU 都保存：
> 1. **完整模型参数**（推理需要）
> 2. **完整梯度**（反向传播需要）
> 3. **完整 Optimizer States**（Adam 需要 fp32 copy + momentum）
> 其中 Optimizer States 占 12×参数大小，是最主要的冗余来源。

**Q2: ZeRO 和模型并行 (MP) 的区别？**
> **答**：
> | | ZeRO | 模型并行 |
> |---|------|---------|
> | 切分对象 | Optimizer/Gradient/Param (状态) | 模型层/参数 (计算图) |
> | 通信时机 | 优化器更新时 | 每层前向/反向时 |
> | 显存节省 | 分片冗余状态 | 切分层参数 |
> | 计算效率 | 接近数据并行 | 可能受通信瓶颈 |
> | 适用 | 模型能放进单卡，但优化器太大 | 模型本身太大 |
> 
> **可以结合**：Megatron-DeepSpeed 同时使用 TP + ZeRO。

**Q3: ZeRO-3 的通信量为什么比 ZeRO-1 大？**
> **答**：
> - ZeRO-1：参数和梯度在每张卡完整保存，只有优化器状态分片
>   → 通信主要在优化器更新时 AllGather states
> - ZeRO-3：参数也分片了
>   → 每次前向需要 AllGather 参数
>   → 每次反向需要 AllGather 参数 + Reduce-Scatter 梯度
>   → 通信量约 2-3× ZeRO-1
> > 但 ZeRO-3 可以训练更大的模型，是 trade-off。

### 计算题

**Q4: 用 Adam 训练一个 13B 模型，batch_size=4，sequence_length=2048，估算 ZeRO-2 下 8 张 A100 (80GB) 能否放下？**
> **答**：
> ```
> 模型参数 (fp16): 2 × 13B = 26 GB
> 梯度 (fp16): 2 × 13B = 26 GB
> Optimizer States (fp32): 12 × 13B = 156 GB
> 
> ZeRO-2 单卡显存:
> = 参数 + 梯度/N + Optimizer/N + 激活值
> = 26 + 26/8 + 156/8 + 激活值
> = 26 + 3.25 + 19.5 + 激活值
> = 48.75 GB + 激活值
> 
> 激活值估算 (activation checkpointing):
> ≈ 2 × batch × seq × hidden × layers
> ≈ 2 × 4 × 2048 × 5120 × 40 × 2B
> ≈ 12.8 GB
> 
> 总计 ≈ 62 GB < 80 GB
> → 可以放下！
> ```

**Q5: ZeRO-3 下 64 张 GPU 训练 175B 模型，单卡显存多少？**
> **答**：
> ```
> 单卡显存 = 16P / N = 16 × 175B / 64 = 43.75 GB
> 加上激活值和 buffer，约 50-60 GB
> → 需要 A100 (80GB) 或 H100
> ```

### 对比分析题

**Q6: DeepSpeed ZeRO vs PyTorch FSDP？**
> **答**：
> - **FSDP** (Fully Sharded Data Parallel) 是 PyTorch 原生的 ZeRO-3 实现
> - **功能对比**：
>   | 特性 | DeepSpeed | FSDP |
>   |------|-----------|------|
>   | ZeRO-1/2/3 | ✅ | 主要是 3 |
>   | Offload | ✅ CPU/NVMe | ✅ CPU |
>   | MoE | ✅ 原生 | ❌ |
>   | 3D Parallelism | ✅ | ❌ |
>   | 易用性 | 需 config | PyTorch 原生 |
> - **选择**：PyTorch 用户优先 FSDP，需要 MoE 或极致优化选 DeepSpeed。

**Q7: 什么情况下 ZeRO-Offload 比纯 GPU 慢很多？**
> **答**：
> 1. **优化器更新频繁**：Adam 每一步都要从 CPU 加载 states
> 2. **CPU 内存带宽不足**：PCIe 带宽 (~32 GB/s) 远小于 HBM (~2 TB/s)
> 3. **NVMe 更慢**：SSD 带宽 (~7 GB/s) 比 CPU 内存还慢
> 4. **小 batch**：无法掩盖通信延迟
> → 通常慢 2-5×，但可以在单卡上训练原本不可能的模型。

### 场景设计题

**Q8: 设计一个训练 70B 模型的方案，预算 32 张 A100 (40GB)。**
> **答**：
> ```
> 需求分析:
> - 70B 模型，Adam，fp16
> - 显存需求: 16 × 70B = 1120 GB
> - 可用显存: 32 × 40 = 1280 GB
> - 很紧张，需要优化
> 
003e 方案:
> 1. ZeRO-3: 单卡参数 = 1120/32 = 35 GB
> 2. Activation Checkpointing: 激活值减半
> 3. Gradient Accumulation: 减小 micro_batch
> 4. 3D Parallelism:
>    - TP=4 (单节点内)
>    - PP=4 (跨节点)
>    - DP=2 (数据并行)
>    - 总计: 4×4×2 = 32 张卡
> 
003e 备选:
> - 如果还是不够，开启 CPU Offload
> - 或使用 DeepSpeed-MoE 减少计算
> ```

**Q9: 训练过程中发现 GPU 显存波动很大，可能原因和解决方案？**
> **答**：
> **原因**：
> 1. 不同长度序列导致激活值大小不同
> 2. ZeRO-3 的参数 AllGather 不是均匀的
> 3. 有 memory fragmentation
> 
003e **解决**：
003e 1. 设置 `gradient_accumulation_steps` 稳定 batch
003e 2. 使用 `contiguous_gradients: true`
003e 3. 设置 `round_robin_gradients: true`
003e 4. 使用 `memory_efficient_linear: true`
003e 5. 监控显存：`see_memory_usage()`

**Q10: DeepSpeed 中 fp16 和 bf16 怎么选？**
> **答**：
> | | fp16 | bf16 |
> |---|------|------|
> | 指数位 | 5 | 8 |
> | 精度 | 高 | 中 |
003e | 动态范围 | 小 (易溢出) | 大 (更稳定) |
003e | 训练稳定性 | 需 Loss Scaling | 通常不需要 |
003e | A100 支持 | ✅ | ✅ |
003e | V100 支持 | ✅ | ❌ |
003e 
003e **建议**：
003e - A100/H100: 优先 bf16（更稳定，不需要 loss scaling）
003e - V100: 只能用 fp16
003e - 如果 fp16 训练不稳定（loss nan），切换到 bf16

---

## 参考资料

- [ZeRO 论文](https://arxiv.org/abs/1910.02054)
- [ZeRO-Infinity 论文](https://arxiv.org/abs/2104.07857)
- [DeepSpeed 官方文档](https://www.deepspeed.ai/)
- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
- [Megatron-DeepSpeed 教程](https://github.com/microsoft/Megatron-DeepSpeed)
- [PyTorch FSDP 文档](https://pytorch.org/docs/stable/fsdp.html)
- [HuggingFace DeepSpeed Integration](https://huggingface.co/docs/transformers/main_classes/deepspeed)

---

*下一步：尝试用 DeepSpeed ZeRO-2 微调一个 7B 模型，观察显存和速度变化。*
