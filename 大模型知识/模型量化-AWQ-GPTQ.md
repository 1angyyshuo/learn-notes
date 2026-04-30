---
title: 模型量化：AWQ 与 GPTQ
date: 2026-04-30
tags: [量化, vllm, awq, gptq, 推理优化, 大模型部署]
---

# 模型量化：AWQ 与 GPTQ

## 概述

**模型量化**是将大语言模型（LLM）的高精度权重（通常是 FP16/BF16）转换为低精度表示（如 INT4/INT8）的技术。通过降低显存占用和提升计算吞吐量，量化使得在消费级 GPU 上部署大模型成为可能。

vLLM 支持多种量化后端，其中 **GPTQ** 和 **AWQ** 是最常用的两种 4-bit 量化方案。

| 特性 | GPTQ | AWQ |
|------|------|-----|
| 量化粒度 | 逐层（Layer-wise） | 逐通道（Channel-wise） |
| 优化目标 | 最小化权重重构误差 | 保护激活感知的重要权重 |
| 校准数据 | 需要 | 需要 |
| 量化位宽 | 4-bit | 4-bit |
| 典型速度提升 | 2-3x | 2-3x |
| 显存节省 | ~75% | ~75% |
| 精度损失 | 中等 | 较低 |

---

## 1. GPTQ（General-purpose Post-Training Quantization）

### 1.1 核心思想

GPTQ 基于 **OBQ（Optimal Brain Quantization）** 方法，将量化问题转化为层-wise 的权重重构优化问题：

```
目标：对每一层的权重 W 进行量化，使得输出变化最小

min ||WX - ŴX||²

其中：
- W：原始 FP16 权重
- Ŵ：量化后的 INT4 权重
- X：校准数据通过该层前的激活值
```

### 1.2 关键机制

#### 逐层量化与误差补偿
GPTQ 按顺序量化每一层，并将量化误差传播到尚未量化的权重上：

```
量化流程（单层内）：
1. 对权重矩阵的某一列进行量化
2. 计算量化误差：error = W_orig - W_quant
3. 将误差补偿到剩余的未量化列上
4. 重复直到所有列量化完成
```

#### 分组量化（Group-wise）
- 将权重按列分成若干组（默认 128 列一组）
- 每组独立计算缩放因子（scale）和零点（zero point）
- 平衡量化精度和额外参数开销

#### 对称 vs 非对称量化
| 模式 | 说明 | 适用场景 |
|------|------|----------|
| 对称 | zero_point = 0，仅 scale | 默认推荐，速度更快 |
| 非对称 | 独立的 scale 和 zero_point | 权重分布不均匀时 |

### 1.3 在 vLLM 中使用

```python
from vllm import LLM

# 方法 1：加载预量化模型（推荐）
llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",  # HuggingFace 上的 GPTQ 模型
    quantization="gptq",
    dtype="auto",
)

# 方法 2：使用 AutoGPTQ 自行量化
# 先安装：pip install auto-gptq
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,  # desc_act=False 对推理更友好
)

model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantize_config=quantize_config,
)
model.quantize(calibration_dataset)
model.save_quantized("./Llama-2-7B-GPTQ")
```

### 1.4 优缺点

**优点**：
- 通用性强，适用于几乎所有 Transformer 架构
- 社区生态成熟，HuggingFace 上有大量预量化模型
- 量化速度快（几分钟到几十分钟）

**缺点**：
- 对异常值（outliers）敏感，可能导致精度下降
- desc_act=True 时推理速度较慢（需激活重排）
- 同等位宽下精度通常不如 AWQ

---

## 2. AWQ（Activation-aware Weight Quantization）

### 2.1 核心思想

AWQ 基于一个关键观察：**并非所有权重的贡献都是相等的**。那些与大幅激活值相乘的权重对模型输出影响更大，应该被"保护"起来。

```
核心洞察：
- 激活值 X 中有少数通道的值特别大（outliers）
- 与这些 outlier 激活相乘的权重对输出影响更大
- 量化时应保留这些重要权值的精度

量化公式：
Ŵ = round(W / s) × s

其中缩放因子 s 的选择会考虑对应的激活幅度：
s = argmin ||(W × X) - (Ŵ × X)||²
```

### 2.2 关键机制

#### 激活感知保护
AWQ 通过分析校准数据的激活分布，识别重要权重通道：

```python
# 伪代码：识别重要权重
for each channel i:
    # 计算该通道激活值的平均幅度
    activation_scale[i] = mean(abs(X[:, i]))
    
    # 激活幅度越大，对应的权重越重要
    importance[i] = activation_scale[i]
```

#### 逐通道缩放（Per-channel Scaling）
为了不直接保留 FP16 权重（会破坏统一量化），AWQ 采用**缩放-量化-还原**策略：

```
1. 对重要权重通道乘以缩放因子 α（如 α=2）
2. 对激活的对应通道除以 α
3. 进行统一量化
4. 推理时缩放相互抵消，但量化误差被重新分配

数学上等价于：
- 权重：W' = W × α  →  量化误差分布在更大数值上，相对误差减小
- 激活：X' = X / α  →  对应激活变小，乘积不变
```

#### 最优缩放因子搜索
AWQ 通过小规模网格搜索确定最优缩放参数：

```python
# 伪代码：搜索最优缩放
best_scale = 1.0
best_loss = inf
for s in [0.5, 1.0, 2.0, 4.0]:
    quantize_with_scale(s)
    loss = evaluate(calibration_data)
    if loss < best_loss:
        best_loss = loss
        best_scale = s
```

### 2.3 在 vLLM 中使用

```python
from vllm import LLM

# 方法 1：加载预量化模型（推荐）
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    dtype="auto",
)

# 方法 2：使用 AutoAWQ 自行量化
# 先安装：pip install autoawq
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
)

model.quantize(
    tokenizer=tokenizer,
    quant_config={
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",  # GEMM 或 GEMV
    },
)
model.save_quantized("./Llama-2-7B-AWQ")
```

### 2.4 优缺点

**优点**：
- 精度损失小，通常优于同等位宽的 GPTQ
- 对激活异常值鲁棒
- 支持融合 GEMM 内核，推理延迟低

**缺点**：
- 量化过程比 GPTQ 慢
- 预量化模型选择略少于 GPTQ
- 需要更多校准数据以达到最佳效果

---

## 3. GPTQ vs AWQ 深度对比

### 3.1 原理对比

| 维度 | GPTQ | AWQ |
|------|------|-----|
| 优化视角 | **权重空间**：最小化权重重构误差 | **激活空间**：保护对输出影响大的权重 |
| 数学方法 | 基于海森矩阵的逐层最优量化 | 基于激活幅度的逐通道缩放保护 |
| 误差处理 | 误差补偿到未量化权重 | 通过缩放重新分配量化误差 |
| 对 Outliers | 敏感，需特殊处理 | 天然鲁棒 |

### 3.2 精度对比（典型值）

以 Llama-2-7B 在 WikiText2 上的 PPL 为例：

| 模型 | 精度 |  WikiText2 PPL | 显存占用 |
|------|------|----------------|----------|
| FP16 原始 | 16-bit | 5.12 | 13.0 GB |
| GPTQ-4bit | 4-bit | 5.35 (+4.5%) | 3.8 GB |
| AWQ-4bit | 4-bit | 5.22 (+2.0%) | 3.8 GB |

> AWQ 通常比 GPTQ 保持更高的精度，尤其是在复杂任务（如代码生成、数学推理）上差距更明显。

### 3.3 速度对比

| 场景 | GPTQ | AWQ |
|------|------|-----|
| 预填充（Prefill）| 快 | 快 |
| 解码（Decode）| 中等 | 快（GEMM 优化更好） |
| desc_act=True | 明显变慢 | 不受影响 |
| 批量推理 | 良好 | 更优 |

### 3.4 选型建议

```
选择 GPTQ 当：
- 需要快速获得量化模型（成熟生态，模型多）
- 使用 desc_act=False（默认推荐）
- 对极致精度要求不高，追求通用性

选择 AWQ 当：
- 对模型精度敏感（如代码、推理任务）
- 需要低延迟在线服务
- 激活分布有明显 outliers 的模型
```

---

## 4. vLLM 量化推理最佳实践

### 4.1 参数配置

```python
# AWQ 推荐配置
llm = LLM(
    model="model-awq",
    quantization="awq",
    dtype="auto",           # AWQ 通常用 auto
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)

# GPTQ 推荐配置
llm = LLM(
    model="model-gptq",
    quantization="gptq",
    dtype="auto",
    gpu_memory_utilization=0.9,
)
```

### 4.2 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 量化模型加载失败 | 缺少量化配置文件 | 确认 `quantize_config.json` 存在 |
| 推理结果异常 | 量化参数不匹配 | 检查 `group_size` 和 `bits` 是否一致 |
| 速度不如预期 | 未使用优化内核 | 确认 CUDA 版本兼容，尝试更新 vLLM |
| 显存仍不足 | KV Cache 占用大 | 降低 `gpu_memory_utilization` 或 `max_model_len` |

### 4.3 与其他优化结合

```python
# 量化 + 前缀缓存
llm = LLM(
    model="model-awq",
    quantization="awq",
    enable_prefix_caching=True,
)

# 量化 + 张量并行
llm = LLM(
    model="model-gptq",
    quantization="gptq",
    tensor_parallel_size=2,
)
```

---

## 5. 其他量化方案速览

| 方案 | 位宽 | 特点 | vLLM 支持 |
|------|------|------|-----------|
| FP8 | 8-bit | NVIDIA H100 原生支持，精度高 | 是 |
| INT8 (SmoothQuant) | 8-bit | 平衡精度和速度 | 是 |
| GGUF | 2-8 bit | llama.cpp 生态，CPU/GPU 混合 | 否 |
| BitsAndBytes (NF4) | 4-bit | HuggingFace 集成，即插即用 | 是 |
| Marlin | 4-bit | 最新 GPU 内核优化，速度极快 | 是 |

---

## 关联笔记

- [[vLLM-学习笔记]]
- [[分布式训练基础]]

## 参考资料

- [AWQ 论文: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [GPTQ 论文: GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [AutoAWQ 官方文档](https://github.com/casper-hansen/AutoAWQ)
- [AutoGPTQ 官方文档](https://github.com/PanQiWei/AutoGPTQ)
- [vLLM 量化文档](https://docs.vllm.ai/en/latest/quantization/index.html)
- [TheBloke 量化模型库](https://huggingface.co/TheBloke)
