# 引用的论文原图

来源：Qiayuan Liao et al., *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*, arXiv:2610.28480v1。

- PDF: https://arxiv.org/pdf/2610.28480v1
- TeX source: https://arxiv.org/src/2610.28480v1
- 下载日期：2026-10-30（Asia/Shanghai）。
- arXiv PDF的SHA-256与之前从Zotero读取的全文完全一致；具体校验和见 `figure_sources.json`。

## 图片与出处

| 原图 | 论文页码 | 生成资产 | 引用目的 |
|---|---:|---|---|
| Fig. 2 | 4 | `figures/paper_fig2_training_loop.pdf` | 对齐采样、模型拟合与BPTT更新 |
| Fig. 3 | 7 | `figures/paper_fig3_scaling.pdf` | 解释大规模结果及其不属于该仓库已覆盖实验的边界 |
| Fig. 4 | 7 | `figures/paper_fig4_controlled.pdf` | 对齐LAFAN1目标与horizon/critic设计，区分评估回报和训练回报 |
| Fig. 5 | 8 | `figures/paper_fig5_efficiency.pdf` | 区分模型rollout吞吐和解析梯度带来的样本效率 |

源包中的这些图由TikZ/PGFPlots直接绘制，并无对应的独立PNG。报告使用同版本PDF中的原图主体进行矢量裁切，不重建曲线数据、不改图内文字、不使用栅格截图。英文原图注未嵌入图片，报告另写中文说明并明确标出原图号和论文引用。

裁切采用Pypdf提取区域，再通过Poppler/Cairo重封装可见矢量对象，避免仅修改CropBox后将图外正文隐藏但保留在PDF检索层中。坐标、页码和版本在JSON清单中固定。Fig. 3-5没有栅格图对象；Fig. 2仅保留原图的两个图标资源，文字与流程线仍为矢量。

## 重建

需要 `pypdf` 和Poppler的 `pdftocairo`：

```sh
python fetch_paper_figures.py
# 或使用已下载且校验和匹配的论文PDF
python fetch_paper_figures.py --paper /absolute/path/to/2610.28480v1.pdf
```

脚本校验全文SHA-256、19页总页数、裁切区域、四张单页资产的尺寸和关键图内标签，并检查图外图注/正文没有残留。生成的四个PDF在Git中被忽略，不提交原论文全文、源包或临时截图。交付的LaTeX ZIP包含这四张资产，可直接编译；从Git源码重建时先执行上面的提取步骤。

## 引用与权利边界

原图归论文作者所有，不继承所审读代码仓库的MIT许可。报告以具名出处的局部引用服务于论文和代码讲解，不把它们标记为本次生成的实验结果，也不声称对其另行授权。读图时保留硬件、数据集、归一化和训练/评估口径的限制。
