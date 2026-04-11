# 学习笔记

基于 Foam + Obsidian + GitHub 的学习笔记系统。

## 目录结构

```
10-Projects/      # 项目笔记
20-Areas/         # 学习主题
30-Resources/     # 书籍/课程笔记
40-Archive/       # 归档和模板
50-Daily/         # 每日日志
60-System/        # 系统配置
```

## 快速开始

### 安装 VSCode 插件
- Foam
- Markdown All in One

### 创建笔记
1. 按 `Ctrl+Shift+P` → `Foam: Open Daily Note` 打开今日笔记
2. 使用 `[[笔记名]]` 创建双链
3. 按 `Ctrl+点击` 跳转链接

### 同步到 GitHub
```bash
git add .
git commit -m "更新笔记"
git push
```

## 使用 Claude Code

需要创建笔记时告诉我：
- "帮我创建关于 XXX 的学习笔记"
- "创建今天的学习日志"
- "整理这周的笔记"
