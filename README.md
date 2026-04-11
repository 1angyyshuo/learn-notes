# 📚 我的学习笔记系统

基于 **Foam** + **Obsidian** + **Claude Code** 的 All-in-One 知识管理方案。

## 📁 目录结构 (Johnny Decimal)

```
10-Projects/      # 项目笔记（有明确开始和结束）
  11-Active/      # 进行中项目
  12-Archived/    # 已完成项目

20-Areas/         # 领域笔记（持续维护）
  21-Learning/    # 学习主题
  22-Work/        # 工作相关

30-Resources/     # 参考资料
  31-Books/       # 读书笔记
  32-Courses/     # 课程笔记
  33-Articles/    # 文章收藏

40-Archive/       # 归档
  41-Templates/   # 笔记模板

50-Daily/         # 日常记录
  51-Journal/     # 每日日志

60-System/        # 系统配置
```

## 🚀 快速开始

### 1. 安装 VSCode 插件
打开 VSCode，点击左侧扩展图标，搜索并安装：
- [Foam](https://marketplace.visualstudio.com/items?itemName=foam.foam-vscode) - 双链笔记核心
- [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one) - Markdown 增强
- [Markdown Preview Enhanced](https://marketplace.visualstudio.com/items?itemName=shd101wyy.markdown-preview-enhanced) - 高级预览

或使用快捷键 `Ctrl+Shift+P` → `Extensions: Show Recommended Extensions`

### 2. 使用 Foam 功能
- `Ctrl+Shift+P` → `Foam: Open Daily Note` - 打开每日笔记
- `[[笔记名]]` - 创建双链
- `Ctrl+Shift+P` → `Foam: Show Graph` - 查看知识图谱

### 3. 与 Obsidian 同步
- 在 Obsidian 中打开此文件夹作为 Vault
- 使用 Claude Code `/obsidian` Skill 自动同步

## 📝 笔记规范

### 文件名规范
- 使用英文或中文，避免特殊字符
- 单词间用 `-` 连接
- 示例：`javascript-closure.md`, `闭包笔记.md`

### 双链语法
```markdown
[[另一个笔记]]
[[另一个笔记|显示文字]]
![[嵌入的笔记]]
```

### 标签使用
```markdown
#标签名 或 #标签名/子标签

示例：
#学习 #编程/javascript #待复习
```

### Frontmatter 模板
```yaml
---
title: 笔记标题
date: 2024-01-01
tags: [标签1, 标签2]
category: 分类
status: 进行中 | 已完成 | 已归档
---
```

## 🤖 Claude Code 集成

### 可用 Skills
- `/obsidian` - 同步知识库
- `/docx` - 导出 Word 文档
- `/pdf` - 导出 PDF

### 快捷命令
```bash
# 打开今日笔记
Ctrl+Shift+P → Foam: Open Daily Note

# 创建新笔记
Ctrl+Shift+P → Foam: Create New Note

# 查看图谱
Ctrl+Shift+P → Foam: Show Graph
```

## 📊 知识图谱

Foam 会自动分析所有笔记的双链关系，生成可视化知识图谱。按 `Ctrl+Shift+P` 输入 `Foam: Show Graph` 查看。

---

*使用 `Ctrl+Shift+P` → `Foam: Open Daily Note` 开始记录今天！*
