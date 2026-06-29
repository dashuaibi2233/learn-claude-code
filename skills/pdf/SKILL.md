---
name: pdf
description: 处理 PDF 文件：提取文本、创建 PDF、合并文档。适用于用户要求读取 PDF、创建 PDF 或处理 PDF 文件时。
---

# PDF 处理技能

你现在具备 PDF 操作的专业能力。请遵循这些工作流：

## 读取 PDF

**方案 1：快速文本提取（优先）**
```bash
# 使用 pdftotext（poppler-utils）
pdftotext input.pdf -  # 输出到 stdout
pdftotext input.pdf output.txt  # 输出到文件

# 如果没有 pdftotext，尝试：
python3 -c "
import fitz  # PyMuPDF
doc = fitz.open('input.pdf')
for page in doc:
    print(page.get_text())
"
```

**方案 2：逐页读取并查看元数据**
```python
import fitz  # pip install pymupdf

doc = fitz.open("input.pdf")
print(f"Pages: {len(doc)}")
print(f"Metadata: {doc.metadata}")

for i, page in enumerate(doc):
    text = page.get_text()
    print(f"--- Page {i+1} ---")
    print(text)
```

## 创建 PDF

**方案 1：从 Markdown 创建（推荐）**
```bash
# 使用 pandoc
pandoc input.md -o output.pdf

# 使用自定义样式
pandoc input.md -o output.pdf --pdf-engine=xelatex -V geometry:margin=1in
```

**方案 2：用代码生成**
```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

c = canvas.Canvas("output.pdf", pagesize=letter)
c.drawString(100, 750, "Hello, PDF!")
c.save()
```

**方案 3：从 HTML 创建**
```bash
# 使用 wkhtmltopdf
wkhtmltopdf input.html output.pdf

# 或使用 Python
python3 -c "
import pdfkit
pdfkit.from_file('input.html', 'output.pdf')
"
```

## 合并 PDF

```python
import fitz

result = fitz.open()
for pdf_path in ["file1.pdf", "file2.pdf", "file3.pdf"]:
    doc = fitz.open(pdf_path)
    result.insert_pdf(doc)
result.save("merged.pdf")
```

## 拆分 PDF

```python
import fitz

doc = fitz.open("input.pdf")
for i in range(len(doc)):
    single = fitz.open()
    single.insert_pdf(doc, from_page=i, to_page=i)
    single.save(f"page_{i+1}.pdf")
```

## 关键库

| 任务 | 库 | 安装 |
|------|----|------|
| 读取/写入/合并 | PyMuPDF | `pip install pymupdf` |
| 从零创建 | ReportLab | `pip install reportlab` |
| HTML 转 PDF | pdfkit | `pip install pdfkit` + wkhtmltopdf |
| 文本提取 | pdftotext | `brew install poppler` / `apt install poppler-utils` |

## 最佳实践

1. **使用工具前始终检查是否已安装**
2. **处理编码问题**：PDF 可能包含多种字符编码
3. **大型 PDF**：逐页处理，避免内存问题
4. **扫描版 PDF 需要 OCR**：如果文本提取为空，使用 `pytesseract`
