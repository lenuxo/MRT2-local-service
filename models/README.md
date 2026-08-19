# 本地模型数据

使用 UV 管理的独立下载命令：

```bash
uv run mrt-download mrt2_small
uv run mrt-download mrt2_base
uv run mrt-download mrt2_small mrt2_base
```

Magenta 官方下载逻辑会把 MusicCoCa、SpectroStream 等共享资源写入 `resources/`，把 MLX 模型写入 `models/`，并可能在本目录的 `.cache/` 中保留下载缓存。这些生成内容和大文件均已被 Git 忽略。
