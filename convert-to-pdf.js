// Markdown 转 PDF 转换脚本
// 使用方法: node convert-to-pdf.js

const fs = require('fs');
const path = require('path');

// 简单的 Markdown 解析器
function markdownToHTML(md) {
    let html = md;

    // 处理代码块
    html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

    // 处理行内代码
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // 处理标题
    html = html.replace(/^###### (.*$)/gim, '<h6>$1</h6>');
    html = html.replace(/^##### (.*$)/gim, '<h5>$1</h5>');
    html = html.replace(/^#### (.*$)/gim, '<h4>$1</h4>');
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // 处理粗体和斜体
    html = html.replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // 处理引用
    html = html.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');

    // 处理表格
    html = processTables(html);

    // 处理分隔线
    html = html.replace(/^---$/gim, '<hr>');

    // 处理链接
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');

    // 处理段落和换行
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');

    return '<p>' + html + '</p>';
}

function processTables(md) {
    const lines = md.split('\n');
    let result = [];
    let inTable = false;
    let tableContent = [];

    for (let line of lines) {
        if (line.includes('|')) {
            if (!inTable) {
                inTable = true;
                tableContent = [];
            }
            // 跳过分隔行 (---)
            if (!line.match(/^\|[-:\|\s]+\|$/)) {
                tableContent.push(line);
            }
        } else {
            if (inTable) {
                result.push(convertTableToHTML(tableContent));
                inTable = false;
            }
            result.push(line);
        }
    }

    if (inTable) {
        result.push(convertTableToHTML(tableContent));
    }

    return result.join('\n');
}

function convertTableToHTML(rows) {
    if (rows.length === 0) return '';

    let html = '<table>\n';

    // 表头
    const headers = rows[0].split('|').filter(c => c.trim());
    html += '  <thead>\n    <tr>\n';
    headers.forEach(h => {
        html += `      <th>${h.trim()}</th>\n`;
    });
    html += '    </tr>\n  </thead>\n';

    // 表体
    if (rows.length > 1) {
        html += '  <tbody>\n';
        for (let i = 1; i < rows.length; i++) {
            const cells = rows[i].split('|').filter(c => c.trim());
            html += '    <tr>\n';
            cells.forEach(c => {
                html += `      <td>${c.trim()}</td>\n`;
            });
            html += '    </tr>\n';
        }
        html += '  </tbody>\n';
    }

    html += '</table>';
    return html;
}

// 主程序
const inputFile = process.argv[2] || '论文阅读报告_Quest.md';
const outputFile = inputFile.replace('.md', '.html');

if (!fs.existsSync(inputFile)) {
    console.error(`错误: 找不到文件 ${inputFile}`);
    process.exit(1);
}

const mdContent = fs.readFileSync(inputFile, 'utf-8');
const bodyContent = markdownToHTML(mdContent);

const htmlTemplate = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>论文阅读报告</title>
    <style>
        @page {
            size: A4;
            margin: 2.5cm;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: "Segoe UI", "Microsoft YaHei", "SimHei", sans-serif;
            font-size: 11pt;
            line-height: 1.8;
            color: #333;
            max-width: 210mm;
            margin: 0 auto;
            padding: 20px;
        }

        h1 {
            font-size: 18pt;
            color: #1a1a1a;
            border-bottom: 3px solid #2c5aa0;
            padding-bottom: 10px;
            margin: 30px 0 20px 0;
            page-break-after: avoid;
        }

        h2 {
            font-size: 14pt;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            margin: 25px 0 15px 0;
            padding-bottom: 8px;
            page-break-after: avoid;
        }

        h3 {
            font-size: 12pt;
            color: #34495e;
            margin: 20px 0 10px 0;
            page-break-after: avoid;
        }

        h4, h5, h6 {
            font-size: 11pt;
            color: #444;
            margin: 15px 0 8px 0;
        }

        p {
            margin: 10px 0;
            text-align: justify;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 10pt;
            page-break-inside: avoid;
        }

        th, td {
            border: 1px solid #bdc3c7;
            padding: 8px 10px;
            text-align: left;
            vertical-align: top;
        }

        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }

        tr:nth-child(even) {
            background-color: #f8f9fa;
        }

        code {
            font-family: Consolas, "Courier New", monospace;
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
            font-size: 10pt;
        }

        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 15px 0;
            page-break-inside: avoid;
        }

        pre code {
            background: none;
            padding: 0;
        }

        blockquote {
            border-left: 4px solid #3498db;
            margin: 15px 0;
            padding: 10px 20px;
            background-color: #ecf0f1;
            font-style: italic;
            page-break-inside: avoid;
        }

        hr {
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }

        ul, ol {
            margin: 10px 0 10px 30px;
        }

        li {
            margin: 5px 0;
        }

        strong {
            color: #2c3e50;
        }

        a {
            color: #3498db;
            text-decoration: none;
        }

        a:hover {
            text-decoration: underline;
        }

        /* 打印优化 */
        @media print {
            body {
                padding: 0;
            }

            h1, h2, h3 {
                page-break-after: avoid;
            }

            table, blockquote, pre {
                page-break-inside: avoid;
            }
        }
    </style>
</head>
<body>
${bodyContent}
</body>
</html>`;

fs.writeFileSync(outputFile, htmlTemplate, 'utf-8');
console.log(`✅ HTML 文件已生成: ${outputFile}`);
console.log('');
console.log('📄 转换为 PDF 的方法:');
console.log('');
console.log('方法1 - 浏览器打印 (推荐):');
console.log('  1. 用 Chrome/Edge 打开生成的 HTML 文件');
console.log('  2. 按 Ctrl+P (或 Cmd+P)');
console.log('  3. 目标打印机选择 "另存为 PDF"');
console.log('  4. 点击保存');
console.log('');
console.log('方法2 - 使用 VSCode 插件:');
console.log('  1. 安装 "Markdown PDF" 插件');
console.log('  2. 打开 Markdown 文件');
console.log('  3. 右键 -> Markdown PDF: Export (pdf)');
console.log('');
console.log('方法3 - 使用 pandoc (需安装):');
console.log('  pandoc "论文阅读报告_Quest.md" -o "论文阅读报告_Quest.pdf" --pdf-engine=xelatex -V CJKmainfont="SimSun"');
