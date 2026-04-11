---
title: Foam 双链笔记系统
date: 2024-04-11
created: 2024-04-11 19:00
tags: [工具, Foam, 笔记方法]
category: 学习方法
status: 已完成
---

# Foam 双链笔记系统

## 概述

**Foam** 是一款基于 VSCode 的**双链笔记**工具，灵感来源于 [Roam Research](https://roamresearch.com/)。它允许你通过 `[[双链]]` 语法创建笔记间的关联，形成**知识图谱**。

## 核心概念

### 什么是双链笔记？

双链笔记（Bidirectional Linking）是一种知识管理方法：
- **正向链接**：在笔记 A 中链接到笔记 B → `[[笔记B]]`
- **反向链接**：笔记 B 自动显示被笔记 A 引用
- **知识图谱**：可视化所有笔记的关联关系

### 与传统文件夹分类的区别

| 传统文件夹 | 双链笔记 |
|-----------|---------|
| 树状层级 | 网状关联 |
| 单一分类 | 多维度关联 |
| 难以发现联系 | 关系自动可视化 |

## Foam 核心功能

### 1. Wiki 链接
```markdown
[[另一个笔记]]              # 基本链接
[[另一个笔记|显示文字]]     # 带自定义显示文字
![[另一个笔记]]             # 嵌入笔记内容
```

### 2. 标签系统
```markdown
#标签名
#编程/javascript
#待复习/高优先级
```

### 3. 每日笔记
- 快捷键：`Ctrl+Shift+P` → `Foam: Open Daily Note`
- 自动生成带日期的笔记文件
- 适合记录每日学习和思考

### 4. 知识图谱
- 命令：`Foam: Show Graph`
- 可视化所有笔记的关联
- 发现知识间的隐藏联系

## 使用技巧

### 创建新笔记的快捷方式
1. 输入 `[[新笔记名]]`
2. 按住 `Ctrl` 点击链接
3. 自动创建并打开新笔记

### 链接自动补全
- 输入 `[[` 后显示所有笔记列表
- 支持模糊搜索

### 图谱导航
- 点击图谱中的节点打开笔记
- 缩放和平移查看整体结构

## 关联笔记

- [[johnny-decimal-分类法]] - 目录分类方法
- [[obsidian-同步配置]] - 与 Obsidian 配合使用
- [[markdown-语法指南]] - Markdown 完整语法

## 参考资料

- [Foam 官方文档](https://foambubble.github.io/foam/)
- [Roam Research](https://roamresearch.com/)
- [Zettelkasten 方法](https://zettelkasten.de/)

---

* Foam 让笔记不再是孤立的文档，而是相互连接的知识网络！*
