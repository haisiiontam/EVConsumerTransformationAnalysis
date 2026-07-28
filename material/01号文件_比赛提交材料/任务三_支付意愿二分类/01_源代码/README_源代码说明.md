# 任务三_支付意愿二分类 — 源代码说明

完整可复现代码位于交付包根目录：

- `../公共源代码/src/` — 三任务实现
- `../公共源代码/main.py` — 入口
- `../公共源代码/scripts/` — 辅助脚本

本目录另附 `infer_task*.py`（若存在）用于加载本任务「训练好的模型」做推理。

复现训练（需自备 `dataSetB.csv`）：

```bash
python main.py --task all --data dataSetB.csv
```

注意：禁止使用测试集标签训练或调参；本仓库内部验证集由冻结 split 产生。
