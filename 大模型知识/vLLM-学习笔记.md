---
title: vLLM 学习笔记
date: 2026-04-30
tags: [vllm, 推理优化, 大模型部署]
category: 大模型工程
status: 进行中
---

# vLLM 学习笔记

## 概述

**vLLM** 是一个高吞吐量、低延迟的大语言模型推理和服务引擎，由伯克利大学开发。核心创新是 **PagedAttention** 算法，通过借鉴操作系统虚拟内存和分页机制，高效管理 KV Cache。

GitHub: https://github.com/vllm-project/vllm

---

## 1. 核心问题：KV Cache 内存管理

### 1.1 传统推理的内存瓶颈

在自回归生成过程中，模型需要缓存之前所有 token 的 Key 和 Value（KV Cache）：

```
生成第 N 个 token 时：
- 需要加载 token 1 到 token N-1 的 KV Cache
- KV Cache 大小 = 2 × num_layers × num_heads × head_dim × seq_len × batch_size
- 序列越长，KV Cache 占用的 GPU 显存越大
```

### 1.2 现有方案的缺陷

| 方案 | 问题 |
|------|------|
| 连续内存分配 | 每个序列预分配最大长度，内部碎片严重 |
| 动态扩容 | 需要内存拷贝，产生额外开销 |
| 静态批处理 | 一个序列完成前其他序列需等待，GPU 利用率低 |

**核心矛盾**：KV Cache 需要动态增长，但 GPU 显存要求连续分配。

---

## 2. PagedAttention：核心创新

### 2.1 设计思想

借鉴操作系统的**虚拟内存分页**机制：

| 操作系统 | vLLM |
|---------|------|
| 物理内存分页 | KV Cache 分页 |
| 虚拟地址到物理地址映射 | 逻辑 KV Block 到物理 Block 映射 |
| 按需分配页面 | 按需分配 KV Block |
| 共享内存页（fork） | 共享 KV Block（Copy-on-Write） |

### 2.2 关键概念

#### KV Block
- 固定大小（默认 16 tokens）
- 物理上非连续存储
- 通过 Block Table 记录逻辑到物理的映射

#### Block Table
```python
# 每个序列维护一个 Block Table
block_table = {
    0: physical_block_5,   # 逻辑 block 0 -> 物理 block 5
    1: physical_block_23,  # 逻辑 block 1 -> 物理 block 23
    2: physical_block_7,   # 逻辑 block 2 -> 物理 block 7
}
```

### 2.3 内存优势

```
传统方式：
序列 A (长度 100) + 序列 B (长度 50) + 序列 C (长度 75)
= 预分配 3 × max_len = 大量内部碎片

vLLM PagedAttention：
Block 大小 = 16 tokens
序列 A: 7 blocks (7×16=112, 实际用 100)
序列 B: 4 blocks (4×16=64, 实际用 50)
序列 C: 5 blocks (5×16=80, 实际用 75)
= 几乎无内部碎片，显存利用率 > 90%
```

---

## 3. 安装与基础使用

### 3.1 安装

```bash
# 基础安装
pip install vllm

# 特定 CUDA 版本
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121

# 从源码安装
 git clone https://github.com/vllm-project/vllm.git
 cd vllm
 pip install -e .
```

### 3.2 离线推理（Python API）

```python
from vllm import LLM, SamplingParams

# 1. 加载模型
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,  # GPU 数量
    dtype="auto",            # 自动选择 float16/bfloat16
    gpu_memory_utilization=0.9,  # GPU 显存使用率
)

# 2. 设置采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256,
)

# 3. 批量推理
prompts = [
    "The future of AI is",
    "Once upon a time",
    "In the field of machine learning,",
]

outputs = llm.generate(prompts, sampling_params)

# 4. 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}")
    print("-" * 50)
```

### 3.3 在线服务（OpenAI-compatible API）

```bash
# 启动服务
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --port 8000 \
    --tensor-parallel-size 1

# 支持参数
# --max-model-len 4096        # 最大序列长度
# --gpu-memory-utilization 0.9 # GPU 显存使用率
# --quantization awq          # 量化方式（awq/gptq）
```

客户端调用：
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    temperature=0.7,
    max_tokens=256,
)

print(response.choices[0].message.content)
```

---

## 4. 高级特性

### 4.1 连续批处理（Continuous Batching）

传统批处理：一个序列完成后才能开始新序列，造成 GPU 空闲。

vLLM 的连续批处理：
```
时间线：
t0: [序列A生成token_1, 序列B生成token_1, 序列C生成token_1]
t1: [序列A完成!, 序列B生成token_2, 序列C生成token_2, 序列D加入]
t2: [序列B生成token_3, 序列C完成!, 序列D生成token_1, 序列E加入]
```

**效果**：GPU 利用率接近 100%，吞吐量提升 10-20 倍。

### 4.2 前缀缓存（Prefix Caching）

```python
# vLLM v0.4.0+ 支持
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_prefix_caching=True,  # 启用前缀缓存
)

# 共享系统提示的场景
system_prompt = "You are a helpful assistant."
prompts = [
    f"{system_prompt}\nUser: What is AI?\nAssistant:",
    f"{system_prompt}\nUser: Explain ML.\nAssistant:",
]
# 系统提示部分只需计算一次 KV Cache，后续复用
```

### 4.3 张量并行与流水线并行

```python
# 张量并行（多 GPU）
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,  # 4 张 GPU
)

# 流水线并行 + 张量并行
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    pipeline_parallel_size=2,  # 共 4×2=8 张 GPU
)
```

### 4.4 量化推理

```python
# AWQ 量化（4-bit）
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    dtype="auto",
)

# GPTQ 量化
llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq",
)

# FP8 (H100 支持)
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="fp8",
)
```

---

## 5. 性能优化实践

### 5.1 参数调优

| 参数 | 说明 | 建议 |
|------|------|------|
| `gpu_memory_utilization` | KV Cache 占显存比例 | 0.85-0.95 |
| `max_model_len` | 最大序列长度 | 按实际需求设置 |
| `max_num_seqs` | 最大并发序列数 | 默认即可 |
| `max_num_batched_tokens` | 每批最大 token 数 | 根据 GPU 调整 |
| `block_size` | KV Block 大小 | 默认 16 |

### 5.2 常见场景配置

**场景 1：高吞吐离线批处理**
```python
llm = LLM(
    model="model-name",
    gpu_memory_utilization=0.95,
    max_num_seqs=256,
    max_model_len=4096,
)
```

**场景 2：低延迟在线服务**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model model-name \
    --max-num-seqs 64 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
```

**场景 3：超长上下文**
```python
llm = LLM(
    model="model-name",
    max_model_len=128000,  # 128K 上下文
    rope_scaling={"type": "dynamic", "factor": 8.0},
)
```

---

## 6. 与 Quest 的结合

Quest（Query-aware Sparsity）可以与 vLLM 结合进一步优化长上下文推理：

```
vLLM 解决：KV Cache 显存管理效率
Quest 解决：长序列 Attention 计算稀疏化

结合效果：
- vLLM PagedAttention → 高效存储和加载 KV Cache
- Quest Top-K 选择 → 只加载关键 pages 进行 attention
- 端到端延迟进一步降低
```

---

## 7. 学习路径

1. **理论理解**
   - [x] PagedAttention 原理
   - [ ] Prefix Caching 机制
   - [ ] Speculative Decoding（推测解码）

2. **动手实践**
   - [ ] 本地部署一个 7B 模型
   - [ ] 对比 vLLM vs HuggingFace 推理速度
   - [ ] 使用连续批处理处理高并发请求

3. **深入优化**
   - [ ] 源码阅读：scheduler.py（调度器）
   - [ ] 源码阅读：worker.py（工作进程）
   - [ ] 自定义 Attention 后端

---

## 参考资料

- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM Paper: Efficient Memory Management for LLM Serving](https://arxiv.org/abs/2309.06180)
- [PagedAttention 深度解析](https://zhuanlan.zhihu.com/p/666473258)
- [vLLM vs Text Generation Inference 对比](https://huggingface.co/docs/text-generation-inference/index)
