# Quest 论文证据地图

## 论文基本信息

| 项目 | 内容 |
|------|------|
| 标题 | Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference |
| 会议 | ICML 2024 (PMLR 235) |
| 年份 | 2024 |
| 作者 | Jiaming Tang*, Yilong Zhao*, Kan Zhu, Guangxuan Xiao, Baris Kasikci, Song Han |
| 单位 | 上海交大、MIT、华盛顿大学、NVIDIA |
| 代码 | https://github.com/mit-han-lab/Quest |

---

## 核心 Claim 与证据映射

### Claim 1: Token 关键性高度依赖 Query（核心洞察）

**论文声称**：
> "criticality of a token highly depends on the query"

**证据来源**：
- **图2** (第2页): "A is B. C is D. A is" 的 attention map
  - Query "D" 时，"B" 注意力分数低
  - Query "is" 时，"B" 注意力分数高
- **表1** (第5页): Passkey retrieval 对比
  - H2O/TOVA/StreamingLLM: 1-10% 准确率
  - Quest: 96-99% 准确率

**证据强度**：★★★★★
- 可视化直观展示动态关键性
- 长距离依赖任务定量对比强烈

---

### Claim 2: 不同层稀疏度差异显著

**论文声称**：
> "For the first two layers, the sparsity is below 10%, while for the rest of the layers, the sparsity is larger than 90%"

**证据来源**：
- **图3** (第3页): Query-aware sparsity for each layer
  - 第0-1层: 稀疏度 <10%
  - 第2-30层: 稀疏度 >90%

**关键数字**：
- 前两层：几乎不能压缩
- 其他层：可实现 10× 以上稀疏度

**证据强度**：★★★★★
- Oracle baseline 对比验证
- Quest 估计与 Oracle 高度吻合

---

### Claim 3: Quest 估计方法有效（关键性估计准确性）

**论文声称**：
> Quest 的关键性估计能准确预测真实注意力分数最高的 tokens

**证据来源**：
- **图4** (第3页): Recall rate of tokens with Top-10 attention scores
  - Quest: 召回率接近 100%
  - H2O: 召回率显著更低（基于历史信息，非 query-aware）

**关键数字**：
- Quest 平均召回率：~95%+
- H2O 平均召回率：~70%

**证据强度**：★★★★☆
- 仅展示 10K context 结果
- 未展示其他长度下的召回率变化

---

### Claim 4: Quest 在多种长上下文任务上保持精度

**论文声称**：
> Quest 在语言建模、passkey retrieval、LongBench 等任务上接近无损

**证据来源**：

1. **语言建模 (PG19)** - 图6 (第5页)
   - Quest vs Full Cache 困惑度曲线几乎重合
   - H2O*/TOVA* 在短输入时困惑度偏高

2. **Passkey Retrieval** - 表1 (第5页)
   | 方法 | 10K (budget 64) | 100K (budget 1024) |
   |------|-----------------|-------------------|
   | H2O | 1% | 2% |
   | TOVA | 1% | 2% |
   | StreamingLLM | 1% | 1% |
   | **Quest** | **99%** | **96%** |

3. **LongBench** - 图7 (第6页)
   - 6个数据集：Qasper, HotpotQA, GovReport, TriviaQA, NarrativeQA, MultifieldQA
   - Quest 1K budget ≈ Full Cache performance
   - 基线 4K budget < Quest 1K budget

**证据强度**：★★★★★
- 覆盖多种任务类型
- 定量对比充分

---

### Claim 5: Quest 实现显著加速

**论文声称**：
> "7.03× self-attention speedup, 2.23× end-to-end latency improvement"

**证据来源**：

1. **Kernel 评估** - 图8 (第7页)
   - Attention latency vs sequence length
   - Quest 延迟基本恒定（仅取决于 token budget）

2. **端到端延迟** - 图10 (第8页)
   | Context | FlashInfer | Quest (budget 2048) | Speedup |
   |---------|------------|-------------------|---------|
   | 8K | 10.2ms | 7.5ms | 1.36× |
   | 16K | 17.7ms | 10.5ms | 1.69× |
   | 32K | 29.6ms | 14.6ms | **2.03×** |

   4-bit 量化下：
   | Context | FlashInfer | Quest (budget 2048) | Speedup |
   |---------|------------|-------------------|---------|
   | 32K | 32.5ms | 14.6ms | **2.23×** |

3. **效率对比** (图11, 第8页)
   - Quest vs TOVA 在相同精度下：
     - GovReport: 3.82× 加速
     - TriviaQA: 4.54× 加速

**证据强度**：★★★★★
- 多维度效率评估
- 与强 baseline (FlashInfer) 对比

---

## 方法设计证据链

### 问题定义
- **输入**: Query 向量 Q，KV Cache（L tokens，P pages）
- **目标**: 选择 K 个最可能关键的 pages
- **约束**: 不预丢弃 KV，动态选择，低开销

### 解决方案
1. **Page 粒度管理** (基于 PageAttention)
   - 每页 S tokens
   - 减少内存碎片

2. **关键性估计**
   - 每页维护 Min Key, Max Key
   - Score = Σ max(Q_i × Min_i, Q_i × Max_i)

3. **两阶段执行**
   - Stage 1: 加载 Min/Max，计算 scores，Top-K 选择
   - Stage 2: 加载 Top-K pages，执行 attention

### 理论保证
- Upper bound: Score 始终是 page 内真实注意力分数的上界
- 不会漏选关键 token

---

## 实验设计评估

### 实验充分性

| 维度 | 覆盖情况 | 评价 |
|------|----------|------|
| 任务类型 | 语言建模、检索、QA、摘要 | ★★★★★ 全面 |
| 模型规模 | 7B | ★★★☆☆ 较小，但可扩展 |
| 序列长度 | 32K, 128K | ★★★★☆ 覆盖主要场景 |
| Baseline | H2O, TOVA, StreamingLLM | ★★★★★ 强基线 |
| 消融实验 | 不同 budget、不同层策略 | ★★★★☆ 较充分 |
| 效率评估 | Kernel + 端到端 | ★★★★★ 完整 |

### 潜在证据缺口

1. **更大模型**: 仅在 7B 上测试，13B/70B 效果未知
2. **更长序列**: 未测试 1M tokens 场景
3. **与其他优化结合**: 未与 speculative decoding 结合测试
4. **不同任务类型**: 未测试代码生成、数学推理等

---

## 关键图表清单

### 必须包含（Motivation + 方法 + 实验）

1. **Figure 3** (第3页)
   - 类型: Motivation / 数据分析
   - 内容: 每层稀疏度分析
   - 关键数字: 前两层 <10%，其他层 >90%

2. **Figure 5** (第4页)
   - 类型: 方法 / 系统架构
   - 内容: Quest 两阶段执行流程
   - 关键: Min/Max Key → Score → Top-K → Attention

3. **Figure 7** (第6页)
   - 类型: 实验 / 主要结果
   - 内容: LongBench 6数据集对比
   - 关键: Quest 1K ≈ Full，基线 4K < Quest 1K

### 建议补充

4. **Table 1** (第5页)
   - 内容: Passkey Retrieval 对比
   - 强调: Quest 长距离依赖能力

5. **Figure 10** (第8页)
   - 内容: 端到端延迟对比
   - 强调: 实际部署价值

---

## 批判性评估

### 优势
- 问题洞察深刻（query-aware）
- 方法简洁高效（min/max + top-k）
- 实验验证充分（多任务、多长度）
- 实际加速显著（2.23× 端到端）

### 局限
- 前两层无法稀疏（需加载完整 KV）
- Page size 固定，未探索自适应
- 仅在 7B 模型上验证
- Token budget 需预设，未自适应

### 延伸方向
- 结合 speculative decoding
- 训练轻量级关键性预测器
- 多模态长序列应用
- 自适应 budget 分配

---

*证据地图生成时间: 2024年4月19日*
*用于: 论文阅读报告作业*
