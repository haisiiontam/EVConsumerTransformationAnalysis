# Task1 版本回退

- 稳定版快照：`backups/task1_stable_50662a4.py`（Git `50662a4`）
- 增强版代码：`src/tasks/task1/` 包
- 正式是否合并以 `outputs/task1/task1_final_config.json` 与论文结论为准

回退到稳定单体文件：

```bash
copy backups\task1_stable_50662a4.py src\tasks\task1.py
# 并移除 src\tasks\task1\ 包目录
```
