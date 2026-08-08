# 本地模型数据

运行以下脚本下载模型：

```bash
../scripts/download_models.sh mrt2_small
../scripts/download_models.sh mrt2_base
../scripts/download_models.sh mrt2_small mrt2_base
```

Magenta 官方 CLI 会将共享资源写入当前目录下的 `resources/`，并将导出的 MLX 模型写入 `models/`。

下载的模型权重体积很大，已经通过项目的 `.gitignore` 排除，不会提交到 Git。
