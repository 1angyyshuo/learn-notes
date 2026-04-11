---
title: Obsidian 同步配置指南
date: 2024-04-11
created: 2024-04-11 19:00
tags: [工具, Obsidian, 同步]
category: 工具配置
status: 已完成
---

# Obsidian 同步配置指南

## 概述

**Obsidian** 是一款强大的本地优先知识库工具。本指南介绍如何将 VSCode + Foam 的笔记与 Obsidian 配合使用，实现最佳的学习笔记体验。

## 配置步骤

### 1. 在 Obsidian 中打开笔记库

1. 打开 Obsidian
2. 点击左下角「打开其他库」
3. 选择你的笔记根目录（如 `e:/learnMarkdown`）
4. 完成！Obsidian 会自动识别所有 Markdown 文件

### 2. 推荐插件

#### 核心插件（Obsidian 内置）
- **图谱 (Graph View)** - 可视化笔记关系
- **反向链接 (Backlinks)** - 查看引用当前笔记的所有笔记
- **标签面板 (Tag Pane)** - 按标签浏览笔记

#### 社区插件推荐
- **Dataview** - 查询笔记元数据
- **Templater** - 高级模板功能
- **Git Integration** - 版本控制
- **Foam VSCode Bridge** - 更好的 VSCode 兼容

### 3. 设置同步

#### 方案一：Git 同步（免费）

1. 初始化 Git 仓库：
```bash
cd ~/learnMarkdown
git init
git add .
git commit -m "Initial commit"
```

2. 创建 GitHub 仓库并推送：
```bash
git remote add origin https://github.com/yourusername/notes.git
git push -u origin main
```

3. 在其他设备上克隆：
```bash
git clone https://github.com/yourusername/notes.git
```

#### 方案二：Obsidian Sync（付费）
- 官方同步服务
- 端到端加密
- 支持版本历史
- $8/月

#### 方案三：iCloud/OneDrive/Dropbox
- 直接将笔记库放在云同步文件夹
- 简单但可能有同步冲突

## Foam + Obsidian 协作

### 文件兼容性

| 功能 | Foam (VSCode) | Obsidian | 兼容性 |
|-----|---------------|----------|--------|
| Wiki 链接 `[[` | ✅ | ✅ | ✅ 完美 |
| 标签 `#tag` | ✅ | ✅ | ✅ 完美 |
| Frontmatter | ✅ | ✅ | ✅ 完美 |
| 嵌入 `![[` | ✅ | ✅ | ✅ 完美 |
| 路径引用 | 相对路径 | 相对路径 | ✅ 完美 |

### 工作流建议

```
VSCode (Foam)          Obsidian
     ↓                      ↑
  创建/编辑笔记 ───────→  查看/回顾
  代码片段记录           知识图谱浏览
  快速双链               深度思考
```

### 最佳实践

1. **编辑在 VSCode**：利用代码编辑器的强大功能
2. **回顾在 Obsidian**：利用图谱和美观的界面
3. **双向同步**：所有更改自动同步

## Claude Code 集成

### 使用 /obsidian Skill

Claude Code 可以通过 `/obsidian` skill 自动管理 Obsidian 笔记：

```
/obsidian
- 扫描项目结构
- 自动生成学习笔记
- 更新知识库索引
```

### 自动工作流

1. 在 VSCode 中编写代码/笔记
2. Claude Code 分析内容
3. 自动创建相关学习笔记
4. 同步到 Obsidian 知识库

## 常见问题

### Q: 图片路径不兼容？
A: 使用相对路径 `./images/pic.png`，两者都支持

### Q: 插件冲突？
A: 禁用重复功能的插件，如只用 Foam 或 Obsidian 的图谱功能

### Q: 同步冲突？
A: 使用 Git 并养成频繁提交的习惯

## 关联笔记

- [[foam-双链笔记]] - Foam 双链系统介绍
- [[claude-code-工作流]] - Claude Code 配合方法

## 参考资料

- [Obsidian 官网](https://obsidian.md/)
- [Obsidian 中文社区](https://obsidian.md/)

---

*VSCode 编辑 + Obsidian 浏览 = 完美的知识工作流*
