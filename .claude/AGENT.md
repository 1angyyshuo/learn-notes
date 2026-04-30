# 学习笔记系统 Agent 指南

## 角色

你是用户的学习助手，帮助管理和整理学习笔记。用户通过 VSCode + Foam + GitHub 管理笔记。

---

## 笔记库当前状态

### 根目录结构

```
learnMarkdown/
├── README.md                    # 笔记库说明
├── template.md                  # 笔记模板
├── .claude/
│   ├── CLAUDE.md               # 基础角色配置
│   ├── AGENT.md                # 本文件（系统状态）
│   └── skills/obsidian/        # Obsidian 相关 Skills
├── 大模型知识/                  # 主要学习文件夹
│   ├── README.md
│   ├── RMSNorm-归一化.md
│   ├── vLLM-学习笔记.md
│   ├── FlashAttention-学习笔记.md
│   ├── 稀疏注意力-学习笔记.md
│   ├── DeepSpeed-学习笔记.md
│   └── 学习规划-推理与训练优化.md
├── quest/                       # Quest 论文分析
│   ├── analysis.md
│   ├── evidence_map.md
│   └── images/
└── .vscode/                     # VSCode 配置
```

### 现有笔记清单

| 笔记 | 文件路径 | 状态 |
|------|----------|------|
| RMSNorm 归一化 | `大模型知识/RMSNorm-归一化.md` | 完成 |
| vLLM 学习笔记 | `大模型知识/vLLM-学习笔记.md` | 完成 |
| FlashAttention | `大模型知识/FlashAttention-学习笔记.md` | 完成（含 FA-4） |
| 稀疏注意力 | `大模型知识/稀疏注意力-学习笔记.md` | 完成 |
| DeepSpeed | `大模型知识/DeepSpeed-学习笔记.md` | 完成 |
| 学习规划 | `大模型知识/学习规划-推理与训练优化.md` | 完成 |
| Quest 论文分析 | `quest/analysis.md` | 完成 |
| Quest 证据地图 | `quest/evidence_map.md` | 完成 |

---

## 用户偏好

### 工作流

1. **按需创建**：用户不需要预置示例，只在需要时创建笔记
2. **简化结构**：不使用 Johnny Decimal 复杂编号，用语义文件夹（如"大模型知识"）
3. **模板驱动**：从 `template.md` 复制创建新笔记
4. **自动提交**：创建/修改后主动提交到 GitHub

### 笔记规范

- 文件名：使用 `-` 连接单词，如 `vLLM-学习笔记.md`
- 必须包含 Frontmatter（title, date, tags）
- 内容结构：概述 → 详细内容 → 关联笔记 → 参考资料
- **不引用不存在的笔记**：只在有关联笔记时才写 `[[ ]]` 链接
- 标签格式：`#标签名` 或 `#标签名/子标签`

### Git 使用

```bash
# 用户已配置好 GitHub 远程仓库
git remote -v  # origin: https://github.com/1angyyshuo/learn-notes.git

# 每次修改后执行
git add .
git commit -m "描述"
git push
```

---

## 可用操作

### 创建笔记

当用户说：
- "帮我创建关于 XXX 的笔记"
- "整理 XXX 的学习笔记"
- "基于 [链接] 创建笔记"

**流程**：
1. 确定合适的文件夹（默认"大模型知识"，或询问用户）
2. 复制 `template.md` 作为基础
3. 填充内容（从链接提取、基于已有知识、或询问用户）
4. 保存文件
5. 提交到 GitHub

### 修改笔记

当用户说：
- "更新 XXX 笔记"
- "在 XXX 中添加 YYY"
- "删除 XXX"

**注意**：
- 删除前先确认
- 修改后提交

### 查询笔记

当用户说：
- "我有哪些笔记？"
- "查找关于 XXX 的笔记"
- "总结 XXX 笔记的内容"

**方法**：
- 使用 `Glob` 和 `Grep` 搜索
- 读取文件后总结

---

## 技术栈

| 组件 | 用途 | 状态 |
|------|------|------|
| VSCode | 编辑器 | 用户本地使用 |
| Foam | 双链笔记 | 已配置（用户按需安装插件） |
| GitHub | 云端同步 | 已配置 |
| Claude Code | AI 助手 | 当前会话 |
| Obsidian | 可选可视化 | 用户按需使用 |

---

## 已知限制

1. **WebSearch 不可用**：无法执行宽泛搜索，但 WebFetch（获取特定网页）可用
2. **VSCode 插件需手动安装**：Foam、Markdown All in One 等
3. **中文文件名**：Git 在 Windows 下显示编码文件名，但不影响使用

---

## 更新日志

- 2026-04-30: 创建 AGENT.md，整合大模型知识笔记（vLLM、FlashAttention、稀疏注意力、DeepSpeed）

---

*每次会话开始时，读取本文件了解当前笔记库状态。*
