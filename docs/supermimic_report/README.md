# SuperMimic 论文与代码对齐报告

本报告审读 arXiv:2610.28480v1 与非官方仓库 `zita-ch/supermimic_vibed` 的固定提交 `8bfc1158bd968f729fcc65d5e897a05b05ea3d36`。

## 文件

- `main.tex` / `sections/`：完整中文 LaTeX 报告；使用 CTeX、Fandol、数学公式、矢量框架图、源码行号链接和对齐矩阵。
- `references.bib`：论文与固定版本仓库引用；论文元数据来自 Zotero，省略本地附件路径。
- `EVIDENCE.md` / `source_manifest.json`：来源、校验和与验证边界。
- `verify_alignment.py`：16项 CPU 验证，依赖 `torch` 与 `numpy`；显式绕过未安装 Isaac Lab 的运动学导入，不验证 FK/物理。

PDF 放在仓库根目录下的 `output/pdf/supermimic_paper_code_alignment_report.pdf`，不纳入 Git。没有修改目标上游代码、当前 OpenHLM 算法代码或 Zotero 条目。

## 编译

需要带中文支持的 XeLaTeX + BibTeX，或可联网获取 TeX bundle 的 Tectonic。可使用 Codex 内置 LaTeX 编辑器打开此目录。标准命令：

```sh
cd /absolute/path/to/OpenHLM/docs/supermimic_report
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
# or
tectonic main.tex --outdir /absolute/path/to/build --keep-logs
```

在本次环境中，内置编译器无法从其 v33 bundle 地址联网获取资源；已改用本地 Tectonic 0.15.0 完成 PDF 编译。PDF 仍然完全由 LaTeX 生成，不是截图/Markdown转PDF，也没有将排版问题掩盖为编译成功。

## 运行验证

在外部位置准备源码，不要把它作为嵌套仓库加入 OpenHLM：

```sh
git clone https://github.com/zita-ch/supermimic_vibed.git /absolute/path/to/source
git -C /absolute/path/to/source checkout 8bfc1158bd968f729fcc65d5e897a05b05ea3d36
python /absolute/path/to/OpenHLM/docs/supermimic_report/verify_alignment.py /absolute/path/to/source
```

本次环境：Python 3.11.18 / PyTorch 2.15.0+cpu / NumPy 2.4.6。结果：16/16通过，另对所有上游 Python 文件做AST解析。未运行仿真、GPU训练、性能复现或实机实验；报告中对论文成绩、仓库自报成绩与本次验证分开表述。

## 文档验收

- PDF共24页；已用Poppler将全部页面渲染并逐页检查，另对框架图、公式页、对齐矩阵、参数表和伪代码页做放大检查。
- 编译无错误、无overfull/underfull box、无缺失字形或未解析引用；保留的字体script提示与禁用shell-escape提示不影响输出。
- PDF结构检查：文本可检索、无页外文字、无异常空白页，包含目录/书签及源码超链接。
- 111处源码行号定位全部通过固定版本文件长度核验；32个上游文本/脚本文件校验和与来源清单一致。
- 相关检查通过后运行 `git diff --check`；源文件和验证材料分单元提交，PDF、ZIP和临时QA图片不纳入Git。
