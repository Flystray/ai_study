# -*- coding: utf-8 -*-
"""
汇总最终报告：合并理论学习 / 论文笔记 / 实验报告 四个部分 + 封面，
生成 报告/完整报告.md，并用 pandoc 转为 Word（报告/完整报告.docx）。
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "..", "报告")
REPORT = os.path.abspath(REPORT)

PARTS = ["01_理论学习.md", "02_论文笔记_蒸馏与生成.md", "03_论文笔记_目标检测.md", "04_实验报告.md", "05_参考文献.md"]

COVER = """---
title: ""
---

# 卷积神经网络（CNN）学习与实验报告

<div align="center">

**主题：基础知识学习 · 2. CNN**

*覆盖内容：卷积神经网络原理与梯度推导、主流 CNN 架构、知识蒸馏、生成式模型（AE / GAN / CGAN / Pix2Pix / CycleGAN）、目标检测（R-CNN 系列 / YOLO / SSD）*

---

</div>

> 说明：本报告包含理论学习、14 篇论文阅读笔记与全部实验（Python + PyTorch 实跑）三部分，
> 实验代码位于 `code/` 目录，全部实验数据与图表位于 `results/` 目录，可复现。

\\newpage

"""

def main():
    lines = [COVER]
    for p in PARTS:
        path = os.path.join(REPORT, p)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        lines.append(content)
        lines.append("\n\\newpage\n")

    md_path = os.path.join(REPORT, "完整报告.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("已生成:", md_path)

    # pandoc 转 docx（数学公式转 Word 原生公式）
    import pypandoc
    docx_path = os.path.join(REPORT, "完整报告.docx")
    pypandoc.convert_file(
        md_path, "docx", outputfile=docx_path,
        extra_args=["--toc", "--toc-depth=2", "-M", "lang=zh-CN",
                    "-f", "markdown+tex_math_dollars+pipe_tables+raw_tex"],
    )
    print("已生成:", docx_path)

if __name__ == "__main__":
    main()
