#!/usr/bin/env python3
"""Convert report_zh.md to PDF with Chinese font support."""

import re
import markdown
from weasyprint import HTML

MD_PATH = "report/report_zh.md"
PDF_PATH = "report/DeepseekV4建模分析报告.pdf"

CSS = """
@page {
    size: A4;
    margin: 2cm 1.8cm;
    @bottom-center { content: counter(page); font-size: 9pt; color: #888; }
}
body {
    font-family: "Noto Sans CJK SC", "Noto Sans SC", sans-serif;
    font-size: 11pt;
    line-height: 1.8;
    color: #222;
}
h1 { font-size: 20pt; text-align: center; margin-top: 1.5cm; margin-bottom: 0.8cm; color: #1a1a2e; }
h2 { font-size: 15pt; margin-top: 1cm; border-bottom: 2px solid #2c3e50; padding-bottom: 4px; color: #2c3e50; page-break-after: avoid; }
h3 { font-size: 13pt; margin-top: 0.7cm; color: #34495e; page-break-after: avoid; }
h4 { font-size: 11.5pt; margin-top: 0.5cm; color: #34495e; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 0.5cm 0; font-size: 9.5pt; }
th, td { border: 1px solid #bbb; padding: 5px 8px; text-align: left; }
th { background: #2c3e50; color: white; font-weight: bold; }
tr:nth-child(even) { background: #f4f6f7; }
td strong { color: #c0392b; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 9pt; background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }
pre { background: #f5f5f5; padding: 10px; border-radius: 4px; font-size: 9pt; overflow-x: auto; }
blockquote { border-left: 3px solid #3498db; padding-left: 12px; color: #555; }
hr { border: none; border-top: 1px solid #ddd; margin: 0.8cm 0; }

/* --- List styling --- */
ul, ol {
    margin: 0.3cm 0;
    padding-left: 2em;
}
ul { list-style-type: disc; }
ul ul { list-style-type: circle; padding-left: 1.8em; }
ol { list-style-type: decimal; }
li {
    margin-bottom: 0.15cm;
    padding-left: 0.3em;
    line-height: 1.7;
}
li > p { margin: 0; }
li strong { color: #2c3e50; }
"""


def fix_list_breaks(md_text: str) -> str:
    """Insert blank line before list items that directly follow a non-list line.

    Python markdown requires a blank line before a list block. Chinese docs
    often write ``段落文字：\\n- 项目一`` without the blank line, which causes
    the list to render inline as a paragraph.
    """
    lines = md_text.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i > 0 and re.match(r"^[\-\*]\s+", line):
            prev = lines[i - 1]
            if prev.strip() and not re.match(r"^[\-\*]\s+", prev) and not re.match(
                r"^\s*$", prev
            ):
                result.append("")  # insert blank line
        result.append(line)
    return "\n".join(result)


with open(MD_PATH, "r", encoding="utf-8") as f:
    md_text = f.read()

md_text = fix_list_breaks(md_text)

html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{html_body}</body>
</html>"""

HTML(string=full_html).write_pdf(PDF_PATH)
print(f"Done: {PDF_PATH}")
