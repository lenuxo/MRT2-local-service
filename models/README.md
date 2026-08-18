# 本地模型数据

使用 Python 下载脚本：

```bash
python ../scripts/download_models.py mrt2_small
python ../scripts/download_models.py mrt2_base
python ../scripts/download_models.py mrt2_small mrt2_base
```

Magenta 官方下载逻辑会把共享资源写入 `resources/`，把 MLX 模型写入 `models/`。这些大文件已被 Git 忽略。
