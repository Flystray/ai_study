# -*- coding: utf-8 -*-
"""
将 报告 目录下的 .md 文件批量转换为浏览器可打开的 .html。
- LaTeX 公式先提取保护，转换后再还原，交由本地 MathJax 渲染
- 输出到 报告/html/ 子目录
"""
import os
import re
import glob
import hashlib
import markdown
from html import escape as html_escape

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "报告")
OUT_DIR = os.path.join(REPORT_DIR, "html")
MATHJAX_SRC = "mathjax/tex-chtml.js"   # 相对 HTML 的引用路径

BLOCK_PH = "@@MJ_BLOCK_{}@@"
INLINE_PH = "@@MJ_INLINE_{}@@"


def protect_math(text):
    """把 $$...$$ 和 $...$ 提取为占位符，避免被 markdown 解析器破坏。"""
    blocks, inlines = [], []

    def _block(m):
        content = m.group(1)
        # 去除 blockquote 引用标记（如 "> \begin{aligned}"），它们不是公式内容
        content = re.sub(r"(?m)^\s*>\s?", "", content)
        blocks.append(content.strip())
        # 前后强制空行，使公式在 markdown 中成为独立段落，便于还原为独立 div
        return "\n\n" + BLOCK_PH.format(len(blocks) - 1) + "\n\n"

    text = re.sub(r"\$\$(.+?)\$\$", _block, text, flags=re.S)

    def _inline(m):
        inlines.append(m.group(1))
        return INLINE_PH.format(len(inlines) - 1)

    text = re.sub(r"\$(?!\$)([^\$\n]+?)(?<!\$)\$", _inline, text)
    return text, blocks, inlines


def restore_math(html, blocks, inlines, toc=""):
    """占位符还原为 MathJax 语法。块级公式套在独立 div 中居中显示。
    toc 里的标题文本含行内公式占位符，需要一并还原。
    公式中的 & < > 需转义为 HTML 实体，否则浏览器会误解析。"""
    for i, f in enumerate(blocks):
        f = html_escape(f, quote=False)
        p = "<p>{}</p>".format(BLOCK_PH.format(i))
        div = '<div class="math-block">\\[{} \\]</div>'.format(f)
        html = html.replace(p, div)
        html = html.replace(BLOCK_PH.format(i), "\\[{} \\]".format(f))
    for i, f in enumerate(inlines):
        f = html_escape(f, quote=False)
        ph = INLINE_PH.format(i)
        html = html.replace(ph, "\\({}\\)".format(f))
        toc = toc.replace(ph, "\\({}\\)".format(f))
    return html, toc


def slugify(value, separator="-", _=None):
    """中文章节无法直接 slugify，用 md5 前缀生成唯一锚点 id。"""
    return hashlib.md5(value.encode("utf-8")).hexdigest()[:8]


TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --main: #2c3e50; --accent: #1a5276; --border: #d5dbe0; --bg: #f6f8fa;
  --code-bg: #f0f2f5; --quote-bar: #b8c4cf;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--main);
  font-family: -apple-system, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
  line-height: 1.75; font-size: 15.5px;
}}
.container {{ max-width: 880px; margin: 32px auto 80px; padding: 40px 56px;
  background: #fff; border: 1px solid var(--border); border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0,0,0,.06); }}
h1 {{ font-size: 26px; color: var(--accent); border-bottom: 3px solid var(--accent);
  padding-bottom: 12px; margin-top: 8px; }}
h2 {{ font-size: 21px; color: var(--accent); border-bottom: 1px solid var(--border);
  padding-bottom: 6px; margin-top: 34px; }}
h3 {{ font-size: 17.5px; color: #34495e; margin-top: 26px; }}
h4 {{ font-size: 16px; color: #566573; }}
a {{ color: var(--accent); }}
p, li {{ text-align: justify; }}
strong {{ color: #1b2631; }}

/* 目录 */
.toc {{ background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
  padding: 14px 22px; margin-bottom: 30px; font-size: 14px; }}
.toc .toc-title {{ font-weight: 700; color: var(--accent); margin-bottom: 6px; }}
.toc ul {{ list-style: none; padding-left: 16px; margin: 4px 0; }}
.toc li {{ margin: 2px 0; }}
.toc a {{ text-decoration: none; }}
.toc a:hover {{ text-decoration: underline; }}

/* 表格 */
table {{ border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 14.5px; }}
th, td {{ border: 1px solid var(--border); padding: 8px 12px; text-align: left; }}
th {{ background: #eef2f6; color: var(--accent); font-weight: 600; }}
tbody tr:nth-child(even) {{ background: #fafbfc; }}

/* 引用块 */
blockquote {{ margin: 16px 0; padding: 10px 18px; background: var(--bg);
  border-left: 4px solid var(--quote-bar); border-radius: 0 4px 4px 0; color: #4a5a66; }}
blockquote p {{ margin: 6px 0; }}

/* 代码 */
code {{ background: var(--code-bg); padding: 2px 6px; border-radius: 4px;
  font-family: Consolas, "Courier New", monospace; font-size: 88%; color: #c0392b; }}
pre {{ background: #f8f9fa; border: 1px solid var(--border); border-radius: 6px;
  padding: 12px 16px; overflow-x: auto; }}
pre code {{ background: none; padding: 0; color: #2c3e50; font-size: 13.5px; }}

/* 数学公式 */
.math-block {{ text-align: center; margin: 20px 0; overflow-x: auto; }}
mjx-container[jax="CHTML"] {{ font-size: 105%; }}
</style>
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] }},
  options: {{ skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }}
}};
</script>
<script src="{mathjax}"></script>
</head>
<body>
<div class="container">
{toc}
<article>
{body}
</article>
</div>
</body>
</html>
"""


def convert_one(src_path):
    with open(src_path, "r", encoding="utf-8") as f:
        raw = f.read()

    text, blocks, inlines = protect_math(raw)

    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
        extension_configs={"toc": {"slugify": slugify, "permalink": False}},
    )
    body = md.convert(text)
    toc_html = md.toc or ""

    body, toc_html = restore_math(body, blocks, inlines, toc_html)

    # 标题取第一个 # 标题
    title_match = re.search(r"^#\s+(.+)$", raw, flags=re.M)
    title = title_match.group(1).strip() if title_match else os.path.splitext(os.path.basename(src_path))[0]

    html = TEMPLATE.format(
        title=title, body=body, toc=toc_html, mathjax=MATHJAX_SRC
    )

    out_path = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(src_path))[0] + ".html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(REPORT_DIR, "*.md")))
    if not files:
        print("未找到 .md 文件")
        return
    for fp in files:
        out = convert_one(fp)
        print("已生成:", out)


if __name__ == "__main__":
    main()
