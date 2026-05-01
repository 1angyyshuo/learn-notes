---
title: vLLM 在 GRPO + LoRA 训练中的适配
date: 2026-05-01
tags: [vllm, grpo, lora, rlhf, 强化学习, verl, deepseek]
category: 大模型工程
status: 完成
---

# vLLM 在 GRPO + LoRA 训练中的适配

## 概述

**GRPO**（Group Relative Policy Optimization）是 DeepSeek-R1 提出的强化学习训练算法，训练过程中需要策略模型不断做**推理采样（rollout）**生成大量响应。vLLM 的高吞吐推理能力天然适合加速这一过程，但训练中模型权重持续更新（LoRA），要求 vLLM 侧的权重同步跟随变化。

**核心问题**：vLLM 是为静态模型设计的推理引擎，GRPO 训练时 LoRA 权重不断变化，如何让两者高效协同？

---

## 1. GRPO 训练流程回顾

```
                    ┌──────────────┐
                    │  Policy Model │  ← LoRA 参数
                    │   (veRL /     │
                    │    FSDP)      │
                    └──────┬───────┘
                           │ forward: 生成 rollout
                    ┌──────▼───────┐
                    │  vLLM Server │  ← 高性能推理引擎
                    │  (rollout)   │
                    └──────┬───────┘
                           │ 返回多个 responses
                    ┌──────▼───────┐
                    │ Reward Model │  ← 打分
                    └──────┬───────┘
                           │ rewards
                    ┌──────▼───────┐
                    │  Train Step  │  ← GRPO loss 更新 LoRA
                    │  (Trainer)   │
                    └──────┬───────┘
                           │ 更新后的 LoRA 权重
                    ┌──────▼───────┐
                    │  同步回 vLLM │  ← 关键适配点
                    └──────────────┘
```

**每轮迭代**：vLLM 用当前 LoRA 做 rollout → 奖励打分 → 训练更新 LoRA → 新 LoRA 同步回 vLLM → 下一轮。

---

## 2. vLLM 需要适配的三个核心问题

### 2.1 权重同步问题

**问题**：训练过程中 LoRA 参数持续更新，vLLM 必须拿到最新参数才能生成正确 rollout。

**解决方案**：

```python
# 典型同步流程（veRL 框架中的实现）

# 1. 训练端：更新 LoRA
trainer.step()  # LoRA_A, LoRA_B 被 optimizer 更新

# 2. 将 LoRA 参数从训练实例同步到 vLLM 实例
for name, param in lora_params.items():
    vllm_engine.update_lora_weight(name, param.data)

# 3. vLLM 内部刷新 CUDA graph / kernel cache
vllm_engine.reset_cache()  # 清掉基于旧权重的编译缓存
```

vLLM 的 `LLMEngine` 提供了 `wake_up` / `update_weight` 接口支持权重热更新。

### 2.2 LoRA Adapter 管理

vLLM 原生支持 LoRA adapter 的动态加载和切换：

```python
from vllm import LLM
from vllm.lora.request import LoRARequest

llm = LLM(model="meta-llama/Llama-2-7b-hf", enable_lora=True)

# 每轮训练后更新 LoRA adapter
lora_request = LoRARequest(
    lora_name="grpo_policy",
    lora_int_id=1,
    lora_path="./checkpoints/step_100/lora_weights",
)

# 推理时指定使用哪个 adapter
outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
```

**GRPO 场景下的管理策略**：

```
方案 A（简单）：每步训练后直接覆盖 adapter 文件，vLLM 重新加载
方案 B（高效）：维护多个 adapter 版本，vLLM 热切换不同 adapter_id
方案 C（最优）：直接内存交换 LoRA 权重，避免磁盘 IO
```

### 2.3 Serving 与 Training 的 GPU 资源分配

GRPO 训练需要同时跑推理（vLLM）和训练（Trainer），有两种部署模式：

#### 模式 A：同卡共享（Colocate）

```
GPU 0: [ vLLM 实例 50% 显存] + [ Trainer 实例 50% 显存]
GPU 1: [ vLLM 实例 50% 显存] + [ Trainer 实例 50% 显存]
...

优点：通信零开销，权重同步快
缺点：显存竞争，可能 OOM
适用：小模型（<7B），LoRA 参数量小
```

#### 模式 B：分离部署（Disaggregated）

```
推理集群：GPU 0-3 跑 vLLM（独立显存）
训练集群：GPU 4-7 跑 Trainer（独立显存）

优点：显存独立，互不干扰
缺点：权重同步需要跨卡通信
适用：大模型（>7B），全量训练
```

**veRL 框架的推荐**：对于 LoRA + 7B 以下模型，使用同卡共享 + GPU 显存精细切分。

---

## 3. veRL 框架的 vLLM 集成

[veRL](https://github.com/volcengine/verl)（Volcano Engine Reinforcement Learning）是字节跳动开源的 LLM RL 训练框架，是目前 vLLM + GRPO 集成的参考实现。

### 3.1 架构

```
veRL 的核心组件：

┌─────────────────────────────────────────────┐
│                  veRL                        │
│                                              │
│  ┌──────────┐    ┌──────────┐              │
│  │ Trainer  │    │  vLLM    │              │
│  │ (FSDP/   │◄──►│  Server  │              │
│  │  Megatron│    │          │              │
│  └──────────┘    └──────────┘              │
│       ▲               │                     │
│       │         ┌─────▼──────┐              │
│       └─────────│  Reward    │              │
│     loss 计算   │  Model     │              │
│                 └────────────┘              │
└─────────────────────────────────────────────┘
```

### 3.2 关键适配点

#### 权重同步机制

```python
# veRL 的 worker 同步逻辑（简化版）

class vLLMRolloutWorker:
    def __init__(self, model_path, lora_config):
        self.vllm = LLM(model=model_path, enable_lora=True)
        self.lora_id = 0
    
    def update_policy(self, lora_state_dict):
        """训练步后同步 LoRA 权重"""
        self.lora_id += 1
        # 保存为临时 checkpoint
        save_path = f"/tmp/lora_step_{self.lora_id}"
        save_lora(lora_state_dict, save_path)
        # 注册为新 adapter
        self.vllm.llm_engine.add_lora(
            lora_name=f"policy_{self.lora_id}",
            lora_path=save_path,
        )
    
    def rollout(self, prompts, sampling_params):
        """用当前 LoRA 做推理采样"""
        lora_req = LoRARequest(
            lora_name=f"policy_{self.lora_id}",
            lora_int_id=self.lora_id,
        )
        return self.vllm.generate(
            prompts, sampling_params, 
            lora_request=lora_req,
        )
```

#### CUDA Graph 优化

vLLM 默认会编译 CUDA Graph 以加速 decode。但训练中 LoRA 权重变化后，旧 CUDA Graph 失效：

```python
# 权重更新后需要重建 CUDA Graph
vllm_config = {
    "enforce_eager": True,        # 开发阶段关掉 CUDA Graph
    "max_num_seqs": 256,
}

# 生产阶段：CUDA Graph + 动态重建
# vLLM 检测到权重更新后自动重建 graph
```

#### 采样参数批量化

GRPO 的 group sampling 需要一次 prompt 生成多条 response（如 group_size=4）：

```python
sampling_params = SamplingParams(
    temperature=1.0,       # GRPO 需要高 temperature 保持探索
    top_p=0.95,
    n=4,                   # 每个 prompt 生成 4 条
    max_tokens=512,
    seed=None,             # 不固定 seed 保证多样性
)
```

---

## 4. GRPO 训练配置示例

以下基于 MiniMax-GRPO 的训练配置（Qwen2.5-3B + LoRA）：

```yaml
# veRL 训练配置
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  lora:
    rank: 64              # LoRA rank
    alpha: 128
    target_modules: 
      - "q_proj"
      - "k_proj"  
      - "v_proj"
      - "o_proj"
      - "gate_proj"
      - "up_proj"
      - "down_proj"

vllm:
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.5    # 给 Trainer 留空间
  max_num_seqs: 64
  max_model_len: 4096
  enforce_eager: true            # 开发阶段关 CUDA Graph
  enable_lora: true
  max_lora_rank: 64

grpo:
  group_size: 4                  # 每个 prompt 生成 4 条
  clip_epsilon: 0.2
  kl_coef: 0.04
  max_prompt_length: 1024
  max_response_length: 2048
  temperature: 1.0

trainer:
  learning_rate: 5e-6
  num_epochs: 3
  micro_batch_size: 4
  gradient_accumulation_steps: 8
```

---

## 5. 常见问题与优化

### 5.1 vLLM 显存占用过高，Trainer OOM

```
排查：
1. 降低 gpu_memory_utilization（如 0.4-0.5）
2. 减少 max_num_seqs，限制 KV Cache
3. 降低 max_model_len
4. 量化 vLLM 端（FP8/AWQ），让出显存给 Trainer
```

### 5.2 权重同步慢，训练等待时间长

```
优化：
1. 使用内存直接交换（NCCL broadcast），避免磁盘 IO
2. 减少同步频率：不是每一步都同步，每 N 步同步一次（累积更新）
3. 异步同步：vLLM 用旧权重继续做批量推理，训练侧不等待
```

### 5.3 rollout 时的采样多样性不足

```
原因：GRPO 需要探索性采样，但 vLLM 默认采样偏保守

解决：
- temperature: 1.0-1.5（需要高探索）
- top_p: 0.9-0.95
- 禁用 repetition_penalty（GRPO 用 KL 惩罚控制偏离）
- 确保 seed 随机化
```

### 5.4 LoRA swap 延迟影响吞吐

vLLM 切换 LoRA adapter 有微小延迟（重排 CUDA Graph），连续大批量推理时建议：

```python
# 按 adapter 分组批量处理
for lora_id, batch in group_by_lora(requests):
    lora_req = LoRARequest(lora_name=f"adapter_{lora_id}")
    outputs = llm.generate(batch, sampling_params, lora_request=lora_req)
```

---

## 关联笔记

- [[vLLM-学习笔记]]
- [[分布式训练基础]]
- [[KV-Cache-详解与面经]]

## 参考资料

- [veRL: Volcano Engine Reinforcement Learning](https://github.com/volcengine/verl)
- [DeepSeek-R1 Paper: GRPO Algorithm](https://arxiv.org/abs/2501.12948)
- [vLLM LoRA Documentation](https://docs.vllm.ai/en/latest/features/lora.html)
- [OpenRLHF: Open-source RLHF Framework](https://github.com/OpenRLHF/OpenRLHF)
- [HuggingFace TRL: GRPO Trainer](https://huggingface.co/docs/trl/en/grpo_trainer)
