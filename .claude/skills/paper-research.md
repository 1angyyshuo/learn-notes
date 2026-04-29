# 论文调研助手

## 描述

辅助用户完成顶会论文阅读报告，针对稀疏注意力(Sparse Attention)主题，自动提取PDF内容、分析论文核心问题、提取关键图表（含原图）、总结系统设计、分析实验结果，最终生成结构化的阅读报告并导出为PDF。

参照 `paper-close-read` 的专业流程，确保产出高质量的图文笔记。

## 调用条件

- 用户说"论文调研"、"分析论文"、"读论文"、"论文报告"
- 用户需要完成论文阅读报告作业
- 用户上传了论文PDF或提供论文信息

## 依赖技能

- `pdf-processing`: 提取PDF文本和表格
- `pdftk-server`: PDF合并和处理
- `pdf-figure-extractor`: 从PDF提取图表原图
- `paper-close-read`: 单篇论文精读流程（证据地图、图文笔记）

## 工作流（一条龙服务）

### Step 1: 准备本地论文材料

参照 `paper-close-read` 的材料准备流程：

```bash
# 创建分析目录
mkdir -p paper_analysis/images

# 复制PDF
cp paper.pdf paper_analysis/source.pdf
```

### Step 2: 提取PDF文本

使用 `pdf-processing` skill：

```python
import pdfplumber

with pdfplumber.open(pdf_path) as pdf:
    # 提取基本信息
    total_pages = len(pdf.pages)
    
    # 提取全文
    full_text = ""
    for page in pdf.pages:
        full_text += page.extract_text() + "\n"
    
    # 保存文本
    with open('paper.txt', 'w', encoding='utf-8') as f:
        f.write(full_text)
```

### Step 3: 提取图表原图

使用 `pdf-figure-extractor` skill：

```python
import fitz  # PyMuPDF

def extract_figure(pdf_path, page_num, fig_number, output_name, y_start=80):
    """
    从PDF提取指定Figure（参照pdf-figure-extractor最佳实践）
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # 搜索Figure caption位置（尝试多种格式）
    search_terms = [f'Figure {fig_number}', f'Fig. {fig_number}']
    fig_rect = None
    
    for term in search_terms:
        text_instances = page.search_for(term)
        if text_instances:
            fig_rect = text_instances[0]
            break
    
    if fig_rect:
        # 图形在caption上方，caption上方留10像素间隙
        y_end = fig_rect.y0 - 10
        rect = fitz.Rect(40, y_start, page.rect.width - 40, y_end)
        
        # 高分辨率渲染 (3x)
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect)
        pix.save(output_name)
        print(f'✓ 已提取: {output_name} ({pix.width}x{pix.height})')
    
    doc.close()
    return fig_rect is not None

# 提取关键图表（3类必须）
extract_figure('source.pdf', 2, 3, 'images/fig3_motivation.png', y_start=120)      # Motivation
extract_figure('source.pdf', 3, 5, 'images/fig5_architecture.png', y_start=100)     # 架构
extract_figure('source.pdf', 5, 7, 'images/fig7_experiments.png', y_start=100)      # 实验
```

#### 图表提取检查清单

- [x] 先定位Caption：`page.search_for(f'Figure {n}')`
- [x] 确定裁剪区域：图形在caption上方，y_end = caption.y0 - 10
- [x] 高分辨率渲染：`matrix=fitz.Matrix(3, 3)`
- [x] 验证图片质量：完整图形、无caption、坐标轴清晰

### Step 4: 建立证据地图（参照paper-close-read）

在写正文前，先建立 `evidence_map.md`：

```markdown
# 论文证据地图

## 核心 Claim 与证据映射

### Claim 1: [核心主张]
**论文声称**: ...
**证据来源**: 
- 图X (第Y页): [内容说明]
- 表Z (第W页): [关键数字]
**证据强度**: ★★★★☆

### Claim 2: ...

## 关键图表清单

| 图表 | 类型 | 来源 | 关键数字 | 用途 |
|------|------|------|----------|------|
| Figure 3 | Motivation | 第3页 | 前两层<10% | 证明分层必要性 |
| Figure 5 | 架构 | 第4页 | 两阶段流程 | 展示方法设计 |
| Figure 7 | 实验 | 第6页 | 1K≈Full | 证明多任务有效性 |

## 实验充分性评估

| 维度 | 覆盖情况 | 评价 |
|------|----------|------|
| 任务类型 | ... | ★★★★★ |
| Baseline | ... | ★★★★☆ |

## 证据缺口
- [缺口1]
- [缺口2]
```

### Step 5: 生成完整图文笔记

参照 `paper-close-read` 的输出模板，创建 `analysis.md`：

```markdown
# 论文阅读报告：[论文标题]

> **作者**: ...
> **会议**: ...
> **年份**: ...

---

## Abstract 原文

> [保留英文摘要原文]

---

## 1. 核心问题与研究动机

### 1.1 问题背景
[背景描述，量化数据用表格呈现]

### 1.2 核心发现
[插入图表]
![图表说明](images/figX.png)
*来源：论文第X页，Figure X*

### 1.3 现有方法局限
[对比表格]

---

## 2. 方法设计

### 2.1 核心思想
### 2.2 关键技术
[插入架构图]
![架构图](images/figY.png)

### 2.3 理论保证
[关键公式用LaTeX]
$$
Score = \sum \max(Q_i \times m_i, Q_i \times M_i)
$$

---

## 3. 实验验证

### 3.1 实验设置
### 3.2 关键结果
[插入实验结果图]
![实验结果](images/figZ.png)

[关键数字表格]

---

## 4. 评价与思考

### 4.1 优势
### 4.2 局限
### 4.3 延伸方向

---

## 总结
[核心贡献表格]

---
*报告生成时间: [date]*
```

### Step 6: 生成图片索引

创建 `images/index.md`：

```markdown
# 图片索引

| 文件名 | 来源 | 内容说明 | 在报告中的用途 |
|--------|------|----------|----------------|
| fig3_motivation.png | 第3页, Figure 3 | ... | 证明分层必要性 |
| fig5_architecture.png | 第4页, Figure 5 | ... | 展示方法设计 |
| fig7_experiments.png | 第6页, Figure 7 | ... | 实验验证 |
```

### Step 7: 导出为PDF

#### 方法1：VSCode + Markdown PDF（推荐）

1. 安装插件 "Markdown PDF"
2. 打开 `analysis.md`
3. 右键 → "Markdown PDF: Export (pdf)"

#### 方法2：pandoc

```bash
pandoc analysis.md -o "学号_姓名_作业2.pdf" \
  --pdf-engine=xelatex \
  -V CJKmainfont="SimSun" \
  -V geometry:margin=2.5cm
```

#### 方法3：浏览器打印

1. VSCode预览 → Open in Browser
2. Ctrl+P → 另存为PDF

### Step 8: 质检清单

交付前核对（参照paper-close-read）：

- [x] Abstract原文是否出现
- [x] 关键claim是否都能回指到图、表、公式或页码
- [x] 方法图和结果图是否都被正文解释，而非只贴上去
- [x] 所有关键数字是否与原文一致
- [x] 是否包含至少3张图表原图
- [x] 图表来源是否都标注（第几页）
- [x] 是否区分"论文明确声称"和"基于证据的解释"
- [x] 是否指出证据缺口或局限

## 快速开始命令

用户可以说：
- "帮我精读这篇论文" → 完整流程（证据地图+图文笔记）
- "提取论文图表" → 提取所有关键图表
- "建立证据地图" → 先建evidence_map.md
- "生成图文报告" → 生成完整报告
- "一条龙服务" → 包含质检的完整流程

## 输出结构

```
paper_analysis/
├── source.pdf              # 原始论文
├── paper.txt               # 提取的文本
├── evidence_map.md         # 证据地图
├── analysis.md             # 完整分析报告
├── images/
│   ├── index.md            # 图片索引
│   ├── fig3_motivation.png # 关键图表
│   ├── fig5_architecture.png
│   └── fig7_experiments.png
└── 学号_姓名_作业2.pdf      # 最终提交文件
```

## 输出要求

- ✅ 体现真实理解，而非简单翻译
- ✅ 行文清晰，逻辑完整
- ✅ 所有图表引用需注明来源（第几页）
- ✅ **必须包含论文原图**，而非文字描述
- ✅ 图片清晰可读，分辨率足够（建议3x渲染）
- ✅ 保留Abstract原文
- ✅ 包含评价与思考
- ✅ 鼓励批判性思考

## 约束检查

- [ ] 论文是2023年及之后发表
- [ ] 是顶会论文（ISCA, ASPLOS, HPCA, MICRO, SOSP, OSDI, ICML, ICLR, NeurIPS等）
- [ ] 主题与稀疏注意力(Sparse Attention)相关
- [ ] 包含至少3张图表原图（motivation、架构、实验）
- [ ] 图表清晰，无caption混入
- [ ] 包含证据地图或等效分析
- [ ] 包含个人评价与思考

## 故障排除

### 图片提取

**问题**: `fitz` 模块找不到  
**解决**: `pip install PyMuPDF`

**问题**: 提取的图片包含caption  
**解决**: 增大 `y_end = fig_rect.y0 - 15`

**问题**: 提取的图片不完整  
**解决**: 减小 `y_start`（如50）

**问题**: 图片分辨率低  
**解决**: 使用 `matrix=fitz.Matrix(3, 3)`

### PDF转换

**问题**: Markdown中的图片不显示  
**解决**: 确保使用相对路径，图片与md同目录

**问题**: pandoc中文字体错误  
**解决**: `--pdf-engine=xelatex -V CJKmainfont="SimSun"`

## 示例

Quest论文完整分析已生成在 `quest_analysis/` 目录：
- `evidence_map.md` - 证据地图
- `analysis.md` - 完整分析报告
- `extract_figures.py` - 图表提取脚本
- `README.md` - 使用说明
